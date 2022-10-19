import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from collections import defaultdict

from config.hparams import multiband_melgan
from vocoder.display import simple_table, stream
from vocoder.wavernn.testset import gen_testset_melgan
from vocoder import base
from vocoder.visualizations_melgan import Visualizations
from vocoder.vocoder_dataset import VocoderDataset, collate_melgan
from vocoder.utils import ValueWindow


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
    generator_loss_window = ValueWindow(100)
    discriminator_loss_window = ValueWindow(100)

    # Let accelerator handle device
    device = accelerator.device

    # Init the model, criterion and optimizers
    try:
        model, _ = base.init_voc_model(model_type, device)
        criterion = base.init_criterion(model_type, device)
        optimizer = base.init_optimizers(model, model_type, device)
    except NotImplementedError as e:
        print(str(e))
        return

    # Initialize the model if not initialized yet
    if force_restart or not weights_fpath.exists():
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("\nStarting the training of {0} from scratch\n".format(model_type))
                save(accelerator, model, weights_fpath, 0, optimizer)

    # Model has been initialized - Load the weights
    print("{0} - Loading weights at {1}".format(device, weights_fpath))
    current_step = (load(model, device, weights_fpath, optimizer) + 1)

    # Determine a couple of params based on model type
    if model_type == base.MODEL_TYPE_MULTIBAND_MELGAN:
        vocoder_hparams = multiband_melgan

    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.json") if ground_truth else voc_dir.joinpath("synthesized.json")
    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    wav_dir = syn_dir.joinpath("wav")
    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir, vocoder_hparams)

    # Init dataloaders
    data_loader = DataLoader(dataset,
                             collate_fn=lambda batch: collate_melgan(batch, vocoder_hparams),
                             batch_size=vocoder_hparams.batch_size,
                             num_workers=threads,
                             shuffle=True,
                             pin_memory=True)
    test_loader = DataLoader(dataset,
                             collate_fn=lambda batch: collate_melgan(batch, vocoder_hparams),
                             batch_size=1,
                             shuffle=True,
                             pin_memory=True)

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    if accelerator.is_local_main_process:
        vis.log_dataset(dataset)
        vis.log_params(vocoder_hparams)

    # Accelerator code - optimize and prepare dataloader, model and optimizer
    data_loader = accelerator.prepare(data_loader)
    model["generator"] = accelerator.prepare(model["generator"])
    model["discriminator"] = accelerator.prepare(model["discriminator"])
    optimizer["generator"] = accelerator.prepare(optimizer["generator"])
    optimizer["discriminator"] = accelerator.prepare(optimizer["discriminator"])

    # Init epoch information
    total_samples = len(dataset)
    gen_finished = False
    dis_finished = False

    # Train until end of
    while not gen_finished and not dis_finished:
        # Get Generator Epoch State
        gen_finished, gen_epoch, gen_epoch_steps, gen_max_step, gen_remaining, gen_loops, gen_init_lr, gen_final_lr = get_epoch_boundaries(
            schedule=vocoder_hparams.generator_tts_schedule,
            model_step=(current_step - vocoder_hparams.generator_train_start_after_steps),
            total_samples=total_samples,
            batch_size=vocoder_hparams.batch_size
        )
        # Get Discriminator Epoch State
        dis_finished, dis_epoch, dis_epoch_steps, dis_max_step, dis_remaining, dis_loops, dis_init_lr, dis_final_lr = get_epoch_boundaries(
            schedule=vocoder_hparams.generator_tts_schedule,
            model_step=(current_step - vocoder_hparams.discriminator_train_start_after_steps),
            total_samples=total_samples,
            batch_size=vocoder_hparams.batch_size
        )

        # Check if finished for early exit of the loop
        if gen_finished or dis_finished:
            # We have completed training. Save the model and exit
            with accelerator.local_main_process_first():
                if accelerator.is_local_main_process:
                    save(accelerator, model, weights_fpath, current_step, optimizer)
            break

        # Determine which Model will finish it's current epoch earlier.
        # This is the boundary for our loop here.
        if gen_max_step > 0 and dis_max_step > 0:
            max_step = min(gen_max_step, dis_max_step)
        else:
            max_step = max(gen_max_step, dis_max_step)

        # Begin the training
        if accelerator.is_local_main_process:
            simple_table([("Epochs (Generator | Discriminator)", "({0} | {1})".format(gen_epoch, dis_epoch)),
                          (f"Remaining Steps in current epoch (Generator | Discriminator)", "({0} | {1}) Steps".format(gen_remaining, dis_remaining)),
                          ('Batch size', vocoder_hparams.batch_size),
                          ('Sequence Len', vocoder_hparams.seq_len)])

        # Calc LR Stepping values
        gen_lr_stepping = (gen_init_lr - gen_final_lr) / np.ceil((total_samples * gen_loops) / vocoder_hparams.batch_size).astype(np.int32)
        dis_lr_stepping = (dis_init_lr - dis_final_lr) / np.ceil((total_samples * dis_loops) / vocoder_hparams.batch_size).astype(np.int32)

        # Training loop
        while current_step < max_step:
            for step, (x, y) in enumerate(data_loader, current_step):
                current_step = step
                start_time = time.time()
                total_train_loss = defaultdict(float)

                # Break out of loop to update training schedule
                if current_step > max_step:
                    # Next epoch
                    break

                # Update lr
                gen_lr = gen_init_lr - (gen_lr_stepping * ((current_step-1) - gen_epoch_steps))
                dis_lr = dis_init_lr - (dis_lr_stepping * ((current_step - 1) - dis_epoch_steps))
                for p in optimizer["generator"].param_groups:
                    p["lr"] = gen_lr
                for p in optimizer["discriminator"].param_groups:
                    p["lr"] = dis_lr

                # Parse batch
                x = tuple([x_.to(device) for x_ in x])
                y = y.to(device)

                # Perform the Training Step
                #######################
                #      Generator      #
                #######################
                if current_step > vocoder_hparams.generator_train_start_after_steps:
                    y_ = model["generator"](*x)

                    # reconstruct the signal from multi-band signal
                    if vocoder_hparams.generator_out_channels > 1:
                        y_mb_ = y_
                        y_ = criterion["pqmf"].synthesis(y_mb_)

                    # initialize
                    gen_loss = 0.0

                    # multi-resolution sfft loss
                    if vocoder_hparams.use_stft_loss:
                        sc_loss, mag_loss = criterion["stft"](y_, y)
                        gen_loss += sc_loss + mag_loss
                        total_train_loss[
                            "train/spectral_convergence_loss"
                        ] += sc_loss.item()
                        total_train_loss[
                            "train/log_stft_magnitude_loss"
                        ] += mag_loss.item()

                    # subband multi-resolution stft loss
                    if vocoder_hparams.use_subband_stft_loss:
                        gen_loss *= 0.5  # for balancing with subband stft loss
                        y_mb = criterion["pqmf"].analysis(y)
                        sub_sc_loss, sub_mag_loss = criterion["sub_stft"](y_mb_, y_mb)
                        gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                        total_train_loss[
                            "train/sub_spectral_convergence_loss"
                        ] += sub_sc_loss.item()
                        total_train_loss[
                            "train/sub_log_stft_magnitude_loss"
                        ] += sub_mag_loss.item()

                    # mel spectrogram loss
                    # if self.config["use_mel_loss"]:
                    #     mel_loss = self.criterion["mel"](y_, y)
                    #     gen_loss += mel_loss
                    #     self.total_train_loss["train/mel_loss"] += mel_loss.item()

                    # weighting aux loss
                    # gen_loss *= self.config.get("lambda_aux", 1.0)

                    # adversarial loss
                    if current_step > vocoder_hparams.discriminator_train_start_after_steps:
                        p_ = model["discriminator"](y_)
                        adv_loss = criterion["gen_adv"](p_)
                        total_train_loss["train/adversarial_loss"] += adv_loss.item()

                        # feature matching loss
                        # if self.config["use_feat_match_loss"]:
                        #     # no need to track gradients
                        #     with torch.no_grad():
                        #         p = self.model["discriminator"](y)
                        #     fm_loss = self.criterion["feat_match"](p_, p)
                        #     self.total_train_loss[
                        #         "train/feature_matching_loss"
                        #     ] += fm_loss.item()
                        #     adv_loss += self.config["lambda_feat_match"] * fm_loss

                        # add adversarial loss to generator loss
                        gen_loss += vocoder_hparams.lambda_adv * adv_loss

                    total_train_loss["train/generator_loss"] += gen_loss.item()

                    # update generator
                    optimizer["generator"].zero_grad()
                    accelerator.backward(gen_loss)
                    if vocoder_hparams.generator_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model["generator"].parameters(),
                            vocoder_hparams.generator_grad_norm,
                        )
                    optimizer["generator"].step()

                #######################
                #    Discriminator    #
                #######################
                if current_step > vocoder_hparams.discriminator_train_start_after_steps:
                    # re-compute y_ which leads better quality
                    with torch.no_grad():
                        y_ = model["generator"](*x)
                    if vocoder_hparams.generator_out_channels > 1:
                        y_ = criterion["pqmf"].synthesis(y_)

                    # discriminator loss
                    p = model["discriminator"](y)
                    p_ = model["discriminator"](y_.detach())
                    real_loss, fake_loss = criterion["dis_adv"](p_, p)
                    dis_loss = real_loss + fake_loss
                    total_train_loss["train/real_loss"] += real_loss.item()
                    total_train_loss["train/fake_loss"] += fake_loss.item()
                    total_train_loss["train/discriminator_loss"] += dis_loss.item()

                    # update discriminator
                    optimizer["discriminator"].zero_grad()
                    accelerator.backward(dis_loss)
                    if vocoder_hparams.discriminator_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model["discriminator"].parameters(),
                            vocoder_hparams.discriminator_grad_norm,
                        )
                    optimizer["discriminator"].step()

                # Update visualizations
                vis.update(total_train_loss, step)

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
                            save(accelerator, model, backup_fpath, current_step, optimizer)

                if save_every != 0 and step % save_every == 0 :
                    # Accelerator: Save in main process after sync
                    accelerator.wait_for_everyone()
                    with accelerator.local_main_process_first():
                        if accelerator.is_local_main_process:
                            print("Saving the model (step %d)" % step)
                            save(accelerator, model, weights_fpath, current_step, optimizer)

                # Evaluate model to generate samples
                # Accelerator: Only in main process
                if accelerator.is_local_main_process and testset_every != 0 and step % testset_every == 0:
                    gen_testset_melgan(accelerator, device, step, model, criterion, test_loader, model_dir, vocoder_hparams)

                # Update Metrics
                time_window.append(time.time() - start_time)
                generator_loss_window.append(total_train_loss["train/generator_loss"])
                discriminator_loss_window.append(total_train_loss["train/discriminator_loss"])

                if accelerator.is_local_main_process:
                    gen_epoch_step = step - gen_epoch_steps
                    gen_epoch_max_step = gen_max_step - gen_epoch_steps
                    dis_epoch_step = step - dis_epoch_steps if dis_epoch > 0 else 0
                    dis_epoch_max_step = dis_max_step - dis_epoch_steps  if dis_epoch > 0 else 0

                    msg = f"| Epoch (Generator | Discriminator): ({gen_epoch} - {gen_epoch_step}/{gen_epoch_max_step} | {dis_epoch} - {dis_epoch_step}/{dis_epoch_max_step}) | LR (Generator | Discriminator): ({gen_lr:#.6}, {dis_lr:#.6}) | Loss (Generator | Discriminator): ({generator_loss_window.average:#.6} | {discriminator_loss_window.average:#.6}) | {1. / time_window.average:#.2}steps/s | Step: {step} |"
                    stream(msg)

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
                save(accelerator, model, backup_fpath, current_step, optimizer)

                # Generate a testset after each epoch
                gen_testset_melgan(accelerator, device, step, model, criterion, test_loader, model_dir, vocoder_hparams)


def save(accelerator, model, path, steps, optimizer=None):
    # Get model type
    model_type = base.get_model_type(model)

    # Unwrap Models
    generator = accelerator.unwrap_model(model["generator"])
    discriminator = accelerator.unwrap_model(model["discriminator"])

    # Build state to be saved
    state = {
        "model_type": model_type,
        "model": {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict()
        },
        "steps": steps
    }

    if optimizer is not None:
        state["optimizer"] = {
            "generator": optimizer["generator"].state_dict(),
            "discriminator": optimizer["discriminator"].state_dict(),
        }

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

    # Load model states
    model["generator"].load_state_dict(checkpoint["model"]["generator"])
    model["discriminator"].load_state_dict(checkpoint["model"]["discriminator"])

    # Load optimizer state
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer["generator"].load_state_dict(checkpoint["optimizer"]["generator"])
        optimizer["discriminator"].load_state_dict(checkpoint["optimizer"]["discriminator"])

    return checkpoint["steps"] if "steps" in checkpoint else 0


def get_epoch_boundaries(schedule, model_step, total_samples, batch_size):
    epoch = 0
    epoch_steps = 0
    max_epoch = len(schedule)
    max_step = 0
    remaining_training_steps = 0
    loops = 1
    init_lr = 1e-3
    final_lr = 1e-3

    for i, session in enumerate(schedule):
        # Schedule not valid yet if model step below 0
        if model_step < 0:
            return False, epoch, epoch_steps, max_step, remaining_training_steps, loops, init_lr, final_lr

        # Update epoch information
        epoch += 1
        epoch_steps = max_step

        # Get session info
        loops, init_lr, final_lr = session

        # Figure out in which Epoch we are
        max_step = np.ceil((total_samples * loops) / batch_size).astype(np.int32) + epoch_steps
        remaining_training_steps = np.ceil(max_step - model_step).astype(np.int32)

        # Do we need to change to the next session?
        if model_step >= max_step:
            # Are there no further sessions than the current one?
            if i == max_epoch - 1:
                # We have completed training.
                return True, epoch, epoch_steps, max_step, remaining_training_steps, loops, init_lr, final_lr
            else:
                # There is a following session, go to it and inc epoch
                continue

        # Schedule not finished
        return False, epoch, epoch_steps, max_step, remaining_training_steps, loops, init_lr, final_lr

    # No Schedule? We have completed training, kinda.
    return True, epoch, epoch_steps, max_step, remaining_training_steps, loops, init_lr, final_lr