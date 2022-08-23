import json
import math
import time
from pathlib import Path
from multiprocessing import Lock

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

from config.hparams import wavernn_fatchord, wavernn_geneing
from vocoder.display import simple_table, stream
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_testset
from vocoder.models import base
from vocoder.visualizations import Visualizations
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.utils import ValueWindow

# global vars
blacklist_lock = Lock()
blacklisted_indices = []

def train(run_id: str, model_type: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool,
          save_every: int, backup_every: int, force_restart: bool,
          vis_every: int, visdom_server: str, no_visdom: bool, testset_every: int, threads: int):

    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir.joinpath(run_id + ".pt")

    # Initialize Accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Accelerator process count: {0}".format(accelerator.num_processes))
        print("Checkpoint path: {}".format(weights_fpath))
        print("Loading training data from: {0} and {1}".format(syn_dir, voc_dir))
        print("Using model: {}".format(model_type))

    # Book keeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # Let accelerator handle device
    device = accelerator.device

    # Init the model
    try:
        model, pruner = base.init_voc_model(model_type, device)
    except NotImplementedError as e:
        print(str(e))
        return

    if model.mode == "MOL":
        loss_func = discretized_mix_logistic_loss
    else:
        # Compared to (N)LL, Cross Entropy is more efficient; see:
        # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
        loss_func = F.cross_entropy

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())

    # Initialize the model if not initialized yet
    if force_restart or not weights_fpath.exists():
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("\nStarting the training of WaveRNN from scratch\n")
                save(accelerator, model, weights_fpath)

    # Model has been initialized - Load the weights
    print("{0} - Loading weights at {1}".format(device, weights_fpath))
    global blacklisted_indices
    blacklisted_indices = load(model, device, weights_fpath, optimizer)
    print("{0} - WaveRNN weights loaded from step {1}".format(device, model.get_step()))

    # Determine a couple of params based on model type
    if model_type == base.MODEL_TYPE_FATCHORD:
        vocoder_hparams = wavernn_fatchord
    elif model_type == base.MODEL_TYPE_GENEING:
        vocoder_hparams = wavernn_geneing

    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.json") if ground_truth else voc_dir.joinpath("synthesized.json")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("wav")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir, vocoder_hparams, blacklisted_indices)
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True)

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    if accelerator.is_local_main_process:
        vis.log_dataset(dataset)
        vis.log_params(vocoder_hparams)
        # FIXME: Print all device names in case we got multiple GPUs or CPUs
        if accelerator.state.num_processes > 1:
            vis.log_implementation({"Devices": str(accelerator.state.num_processes)})
        else:
            device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            vis.log_implementation({"Device": device_name})

    # Init epoch information
    epoch = 0
    max_step = 0

    for i, session in enumerate(vocoder_hparams.voc_tts_schedule):
        # Update epoch information
        epoch += 1
        epoch_steps = max_step

        # Unwrap model after each epoch (if necessary) for re-calibration
        model = accelerator.unwrap_model(model)

        # Get the current step being processed
        current_step = model.get_step() + 1

        # Get session info
        loops, sgdr_init_lr, sgdr_final_lr, batch_size = session

        # Init dataloader
        data_loader = DataLoader(dataset,
                                 collate_fn=lambda batch: collate_vocoder(batch, vocoder_hparams),
                                 batch_size=batch_size,
                                 num_workers=threads,
                                 shuffle=True,
                                 pin_memory=True)

        # Processing mode
        processing_mode = model.mode

        # Accelerator code - optimize and prepare model
        model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

        # Iterate over whole dataset for X loops according to schedule
        total_samples = len(dataset)
        overall_batch_size = batch_size * accelerator.state.num_processes  # Split training steps by amount of overall batch
        max_step = np.ceil((total_samples * loops) / overall_batch_size).astype(np.int32) + epoch_steps
        training_steps = np.ceil(max_step - current_step).astype(np.int32)

        # Calc SGDR values
        sgdr_lr_stepping = (sgdr_init_lr - sgdr_final_lr) / np.ceil((total_samples * loops) / overall_batch_size).astype(np.int32)
        lr = sgdr_init_lr - (sgdr_lr_stepping * ((current_step-1) - epoch_steps))

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(vocoder_hparams.voc_tts_schedule) - 1:
                # We have completed training. Save the model and exit
                with accelerator.local_main_process_first():
                    if accelerator.is_local_main_process:
                        save(accelerator, model, weights_fpath, optimizer)
                break
            else:
                # There is a following session, go to it and inc epoch
                continue

        # Begin the training
        if accelerator.is_local_main_process:
            simple_table([("Epoch", epoch),
                          (f"Remaining Steps in current epoch", str(training_steps) + " Steps"),
                          ('Batch size', batch_size),
                          ("Current init LR", lr),
                          ("LR Stepping", sgdr_lr_stepping),
                          ('Sequence Len', vocoder_hparams.seq_len)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        # Loss anomaly detection
        # If loss change after a step differs by > 50%, this will print info which training data was in the last batch
        avgLossDiff = 0
        avgLossCount = 0
        lastLoss = 0

        # Training loop
        while current_step < max_step:
            for step, (x, y, m, indices) in enumerate(data_loader, current_step):
                current_step = step
                start_time = time.time()

                # Break out of loop to update training schedule
                if current_step > max_step:
                    # Next epoch
                    break

                # Update lr
                lr = sgdr_init_lr - (sgdr_lr_stepping * ((current_step-1) - epoch_steps))
                for p in optimizer.param_groups:
                    p["lr"] = lr

                # Forward pass
                y_hat = model(x, m)
                if processing_mode == 'MOL':
                    y = y.float()
                else:
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                y = y.unsqueeze(-1)

                # Backward pass
                loss = loss_func(y_hat, y)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                # Model Pruning
                if pruner is not None and step >= vocoder_hparams.start_prune:
                    accelerator.wait_for_everyone()
                    with accelerator.local_main_process_first():
                        base_model = accelerator.unwrap_model(model)
                        pruner.update_layers(base_model.prune_layers)
                        num_pruned, z = pruner.prune(base_model.step)
                else:
                    num_pruned, z = 0, torch.FloatTensor([0.0])

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                # anomaly detection
                if avgLossCount == 0:
                    currentLossDiff = 0
                    avgLossDiff = 0
                else:
                    currentLossDiff = abs(lastLoss - loss.item())

                if vocoder_hparams.anomaly_detection and \
                        (step > 5000 and \
                         avgLossCount > 50 and \
                         currentLossDiff > (avgLossDiff * vocoder_hparams.anomaly_trigger_multiplier) \
                         or math.isnan(currentLossDiff) \
                         or math.isnan(loss.item())):  # Give it a few steps to normalize, then do the check
                    print("WARNING - Anomaly detected! (Step {}, Thread {}) - Avg Loss Diff: {}, Current Loss Diff: {}".format(step, accelerator.process_index, avgLossDiff, currentLossDiff))

                    if vocoder_hparams.anomaly_blacklist_batches:
                        # blacklist indices and update dataset
                        global blacklist_lock
                        with blacklist_lock:
                            blacklisted_indices.extend(indices)
                            print("Blacklisting {} indices from dataset. Total blacklisted: {}.".format(len(indices), len(blacklisted_indices)))

                # Comparison to ensure all threads have the same blacklisting
                if len(dataset.get_blacklisted_indices()) != len(blacklisted_indices):
                    dataset.blacklisted_indices(blacklisted_indices)

                # Kill process if NaN
                if math.isnan(loss.item()):
                    currentLossDiff /= 0

                # Update avg loss count
                avgLossDiff = (avgLossDiff * avgLossCount + currentLossDiff) / (avgLossCount + 1)
                avgLossCount += 1

                if accelerator.is_local_main_process:
                    epoch_step = step - epoch_steps
                    epoch_max_step = max_step - epoch_steps
                    if pruner is not None:
                        msg = f"| Epoch: {epoch} ({epoch_step}/{epoch_max_step}) | LR: {lr:#.6} | Loss: {loss_window.average:#.6} | {1. / time_window.average:#.2}steps/s | Step: {step} | Pruned: {num_pruned} ({(round(z.item() * 100, 2))}%) | Currently blacklisted: {len(blacklisted_indices)} |"
                    else:
                        msg = f"| Epoch: {epoch} ({epoch_step}/{epoch_max_step}) | LR: {lr:#.6} | Loss: {loss_window.average:#.6} | {1./time_window.average:#.2}steps/s | Step: {step} | Currently blacklisted: {len(blacklisted_indices)} |"
                    stream(msg)

                lastLoss = loss.item()

                # Update visualizations
                vis.update(loss.item(), step)

                # Save visdom values
                if accelerator.is_local_main_process and vis_every != 0 and step % vis_every == 0:
                    vis.save()

                if backup_every != 0 and step % backup_every == 0:
                    # Accelerator: Save in main process after sync
                    accelerator.wait_for_everyone()
                    with accelerator.local_main_process_first():
                        if accelerator.is_local_main_process:
                            print("Making a backup (step %d)" % step)
                            backup_fpath = Path("{}/{}_{}.pt".format(str(weights_fpath.parent), run_id, step))
                            save(accelerator, model, backup_fpath, optimizer, blacklisted_indices)

                if save_every != 0 and step % save_every == 0 :
                    # Accelerator: Save in main process after sync
                    accelerator.wait_for_everyone()
                    with accelerator.local_main_process_first():
                        if accelerator.is_local_main_process:
                            print("Saving the model (step %d)" % step)
                            save(accelerator, model, weights_fpath, optimizer, blacklisted_indices)

                # Evaluate model to generate samples
                # Accelerator: Only in main process
                if accelerator.is_local_main_process and testset_every != 0 and step % testset_every == 0:
                    eval_model = accelerator.unwrap_model(model)
                    gen_testset(eval_model, test_loader, model_dir, vocoder_hparams)

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

        # Add line break to output and make a backup after every epoch
        # Accelerator: Save in main process after sync
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("")
                print("Making a backup (step %d)" % current_step)
                backup_fpath = Path("{}/{}_{}.pt".format(str(weights_fpath.parent), run_id, current_step))
                save(accelerator, model, backup_fpath, optimizer, blacklisted_indices)

                # Generate a testset after each epoch
                eval_model = accelerator.unwrap_model(model)
                gen_testset(eval_model, test_loader, model_dir, vocoder_hparams)


def save(accelerator, model, path, optimizer=None, blacklisted_indices=[]):
    # Unwrap Model
    model = accelerator.unwrap_model(model)

    # Get model type
    model_type = base.get_model_type(model)

    # Build state to be saved
    state = {
        "model_state": model.state_dict(),
        "model_type": model_type,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if len(blacklisted_indices) > 0:
        print("Current amount of blacklisted dataset indices: %d" % len(blacklisted_indices))
        state["blacklisted_indices"] = blacklisted_indices

    # Save
    torch.save(state, str(path))


def load(model, device, path, optimizer=None):
    # Use device of model params as location for loaded state
    checkpoint = torch.load(str(path), map_location=device)

    # Load and print model type if detected.
    # Here this is mainly for debug reasons; when training you'll need to provide
    # the correct model type anyway, esp. when training initially.
    if "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
        print("Detected model type: %s" % model_type)

    # Load model state
    model.load_state_dict(checkpoint["model_state"])

    # Load optimizer state
    if "optimizer_state" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Load training meta information
    blacklisted_indices = []
    if "blacklisted_indices" in checkpoint and len(checkpoint["blacklisted_indices"]) > 0:
        blacklisted_indices = checkpoint["blacklisted_indices"]
    return blacklisted_indices
