from collections import OrderedDict
import re

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from synthesizer import audio
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.visualizations import Visualizations
from vocoder.display import *
from datetime import datetime
from accelerate import Accelerator
import numpy as np
from pathlib import Path
import sys
import time
import platform


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def train_acc(run_id: str, syn_dir: str, models_dir: str, save_every: int, threads: int,
              backup_every: int, force_restart:bool, vis_every: int, visdom_server: str, no_visdom: bool, hparams):

    syn_dir = Path(syn_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)
    
    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")
    metadata_fpath = syn_dir.joinpath("train.txt")

    # Initialize Accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Accelerator process count: {0}".format(accelerator.state.num_processes))
        print("Checkpoint path: {}".format(weights_fpath))
        print("Loading training data from: {}".format(metadata_fpath))
        print("Using model: Tacotron")
    
    # Book keeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # Let accellerator handle device
    device = accelerator.device

    # Instantiate Tacotron Model
    print("{} - Initialising Tacotron Model...".format(device))
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())

    # Initialize the model if not initialized yet
    if force_restart or not weights_fpath.exists():
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("\nStarting the training of Tacotron from scratch\n")
                save(accelerator, model, weights_fpath)

                # Embeddings metadata
                char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
                with open(char_embedding_fpath, "w", encoding="utf-8") as f:
                    for symbol in symbols:
                        if symbol == " ":
                            symbol = "\\s"  # For visual purposes, swap space with \s

                        f.write("{}\n".format(symbol))

    # Model is has been initialized - Load the weights
    print("{0} - Loading weights at {1}".format(device, weights_fpath))
    load(model, device, weights_fpath, optimizer)
    print("{0} - Tacotron weights loaded from step {1}".format(device, model.get_step()))
    
    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.json")
    mel_dir = syn_dir.joinpath("mels")
    embed_dir = syn_dir.joinpath("embeds")
    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)
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

    epoch = 1
    epoch_steps = 0
    for i, session in enumerate(hparams.tts_schedule):
        # Unwrap model after each epoch (if necessary) for re-calibration
        model = accelerator.unwrap_model(model)

        # Get the current step being processed
        current_step = model.get_step() + 1

        # Fetch params from session
        r, lr, loops, batch_size = session
        
        # Update Model params
        model.r = r

        # Init dataloader
        data_loader = DataLoader(dataset,
                                 collate_fn=lambda batch: collate_synthesizer(batch, r, hparams),
                                 batch_size=batch_size,
                                 num_workers=threads,
                                 shuffle=True,
                                 pin_memory=True)

        # Accelerator code - optimize and prepare model
        model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

        # Iterate over whole dataset for X loops according to schedule
        total_samples = len(dataset)
        overall_batch_size = batch_size * accelerator.state.num_processes # Split training steps by amount of overall batch
        max_step = np.ceil((total_samples * loops) / overall_batch_size).astype(np.int32) + epoch_steps
        training_steps = np.ceil(max_step - current_step).astype(np.int32)

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(hparams.tts_schedule) - 1:
                # We have completed training. Save the model and exit
                with accelerator.local_main_process_first():
                    if accelerator.is_local_main_process:
                        save(accelerator, model, weights_fpath, optimizer)
                break
            else:
                # There is a following session, go to it and inc epoch
                epoch += 1
                epoch_steps = max_step
                continue

        # Begin the training
        if accelerator.is_local_main_process:
            simple_table([("Epoch", epoch),
                          (f"Remaining Steps with r={r}", str(training_steps) + " Steps"),
                          ("Batch Size", batch_size),
                          ("Learning Rate", lr),
                          ("Outputs/Step (r)", r)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        # Training loop
        while current_step < max_step:
            for step, (texts, mels, embeds, idx) in enumerate(data_loader, current_step):
                current_step = step
                start_time = time.time()

                # Generate stop tokens for training
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(idx):
                    stop[j, :int(dataset.metadata[k][4])-1] = 0

                texts = texts.to(device)
                mels = mels.to(device)
                embeds = embeds.to(device)
                stop = stop.to(device)

                # Forward pass
                m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = m1_loss + m2_loss + stop_loss

                optimizer.zero_grad()
                accelerator.backward(loss)

                if hparams.tts_clip_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                    #if np.isnan(grad_norm.cpu()):
                    #    print("grad_norm was NaN!")

                optimizer.step()

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                # Stream update on training progress
                if accelerator.is_local_main_process:
                    epoch_step = step - epoch_steps
                    epoch_max_step = max_step - epoch_steps
                    msg = f"| Epoch: {epoch} ({epoch_step}/{epoch_max_step}) | Loss: {loss_window.average:#.4} | {1./time_window.average:#.2} steps/s | Step: {step} | "
                    stream(msg)

                # Update visualizations
                vis.update(loss.item(), step)

                # Save visdom values
                if accelerator.is_local_main_process and vis_every != 0 and step % vis_every == 0:
                    vis.save()

                # Backup or save model as appropriate
                if backup_every != 0 and step % backup_every == 0:
                    # Accelerator: Save in main process after sync
                    accelerator.wait_for_everyone()
                    with accelerator.local_main_process_first():
                        if accelerator.is_local_main_process:
                            print("Making a backup (step %d)" % step)
                            backup_fpath = Path("{}/{}_{}.pt".format(str(weights_fpath.parent), run_id, step))
                            save(accelerator, model, backup_fpath, optimizer)

                if save_every != 0 and step % save_every == 0:
                    # Accelerator: Save in main process after sync
                    accelerator.wait_for_everyone()
                    with accelerator.local_main_process_first():
                        if accelerator.is_local_main_process:
                            print("Saving the model (step %d)" % step)
                            save(accelerator, model, weights_fpath, optimizer)

                # Evaluate model to generate samples
                # Accelerator: Only in main process
                if accelerator.is_local_main_process:
                    epoch_eval = hparams.tts_eval_interval == -1 and step == max_step  # If epoch is done
                    step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0  # Every N steps
                    if epoch_eval or step_eval:
                        for sample_idx in range(hparams.tts_eval_num_samples):
                            # At most, generate samples equal to number in the batch
                            if sample_idx + 1 <= len(texts):
                                # Remove padding from mels using frame length in metadata
                                mel_length = int(dataset.metadata[idx[sample_idx]][4])
                                mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                                target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                                attention_len = mel_length // r

                                eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                           mel_prediction=mel_prediction,
                                           target_spectrogram=target_spectrogram,
                                           input_seq=np_now(texts[sample_idx]),
                                           step=step,
                                           plot_dir=plot_dir,
                                           mel_output_dir=mel_output_dir,
                                           wav_dir=wav_dir,
                                           sample_num=sample_idx + 1,
                                           loss=loss,
                                           hparams=hparams)

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

        # Add line break after every epoch
        if accelerator.is_local_main_process:
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

def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
