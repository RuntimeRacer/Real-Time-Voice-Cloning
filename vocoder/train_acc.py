import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

import vocoder.hparams as hp
from vocoder.display import simple_table, stream
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_testset
from vocoder.models.fatchord_version import WaveRNN
from vocoder.visualizations import Visualizations
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.utils import ValueWindow


def train_acc(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool,
          save_every: int, backup_every: int, force_restart: bool,
          vis_every: int, visdom_server: str, no_visdom: bool, threads: int):
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Initialize Accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Accelerator process count: {0}".format(accelerator.num_processes))

    # Book keeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # Let accelerator handle device
    device = accelerator.device

    # Instantiate the model
    print("{} - Initializing the model...".format(device))
    model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups: 
        p["lr"] = hp.voc_lr

    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir.joinpath(run_id + ".pt")

    # Initialize the model if not initialized yet
    if force_restart or not weights_fpath.exists():
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("\nStarting the training of WaveRNN from scratch\n")
                save(accelerator, model, weights_fpath)

    # Model has been initialized - Load the weights
    print("{0} - Loading weights at {1}".format(device, weights_fpath))
    load(model, device, weights_fpath, optimizer)
    print("{0} - WaveRNN weights loaded from step {1}".format(device, model.get_step()))
    
    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.json") if ground_truth else voc_dir.joinpath("synthesized.json")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("audio")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
    total_samples = len(dataset)
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True)

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    if accelerator.is_local_main_process:
        vis.log_dataset(dataset)
        vis.log_params()
        # FIXME: Print all device names in case we got multiple GPUs or CPUs
        if accelerator.state.num_processes > 1:
            vis.log_implementation({"Devices": str(accelerator.state.num_processes)})
        else:
            device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            vis.log_implementation({"Device": device_name})

    # Begin the training
    if accelerator.is_local_main_process:
        simple_table([('Batch size', hp.voc_batch_size),
                      ('LR', hp.voc_lr),
                      ('Sequence Len', hp.voc_seq_len)])

    epoch_steps = 0
    for epoch in range(1, 350):
        # Unwrap model after each epoch (if necessary) for re-calibration
        model = accelerator.unwrap_model(model)

        # Get the current step being processed
        current_step = model.get_step() + 1

        # Determine poch stats
        overall_batch_size = hp.voc_batch_size * accelerator.state.num_processes
        max_step = np.ceil((total_samples) / overall_batch_size).astype(np.int32) + epoch_steps

        # Skip here in case this epoch has already been processed
        if current_step > max_step:
            # Update epoch steps
            epoch_steps = max_step
            # Next epoch
            continue

        # Processing mode
        processing_mode = model.mode

        # Init dataloader
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_vocoder,
                                 batch_size=hp.voc_batch_size,
                                 num_workers=threads,
                                 shuffle=True,
                                 pin_memory=True)

        # Accelerator code - optimize and prepare model
        model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

        # Training loop
        for step, (x, y, m) in enumerate(data_loader, current_step):
            current_step = step

            if current_step > max_step:
                # Next epoch
                break

            start_time = time.time()

            #if torch.cuda.is_available():
                #x, m, y = x.cuda(), m.cuda(), y.cuda()
            
            # Forward pass
            y_hat = model(x, m)
            if processing_mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif processing_mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)
            
            # Backward pass
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            time_window.append(time.time() - start_time)
            loss_window.append(loss.item())

            if accelerator.is_local_main_process:
                epoch_step = step - epoch_steps
                epoch_max_step = max_step - epoch_steps
                msg = f"| Epoch: {epoch} ({epoch_step}/{epoch_max_step}) | Loss: {loss_window.average:#.6} | {1./time_window.average:#.2}steps/s | Step: {step} | "
                stream(msg)

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
                        save(accelerator, model, backup_fpath, optimizer)
                
            if save_every != 0 and step % save_every == 0 :
                # Accelerator: Save in main process after sync
                accelerator.wait_for_everyone()
                with accelerator.local_main_process_first():
                    if accelerator.is_local_main_process:
                        print("Saving the model (step %d)" % step)
                        save(accelerator, model, weights_fpath, optimizer)

        # Update epoch steps after training
        epoch_steps = max_step

        # Evaluate model to generate samples
        # Accelerator: Only in main process
        if accelerator.is_local_main_process:
            eval_model = accelerator.unwrap_model(model)
            gen_testset(eval_model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                        hp.voc_target, hp.voc_overlap, model_dir)
        print("")

def save(accelerator, model, path, optimizer=None):
    # Unwrap Model
    model = accelerator.unwrap_model(model)

    # Save
    if optimizer is not None:
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, str(path))
    else:
        torch.save({
            "model_state": model.state_dict(),
        }, str(path))

def load(model, device, path, optimizer=None):
    # Use device of model params as location for loaded state
    checkpoint = torch.load(str(path), map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state"])

    # Load optimizer state
    if "optimizer_state" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])