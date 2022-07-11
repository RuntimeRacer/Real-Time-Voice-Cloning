from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

from synthesizer import audio
from synthesizer.models import base
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import (SynthesizerDataset,
                                             collate_synthesizer)
from synthesizer.utils import ValueWindow
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.visualizations import Visualizations
from vocoder.display import *
from utils.display import *

from config.hparams import tacotron as hp_tacotron, forward_tacotron as hp_forward_tacotron, sp, preprocessing, sv2tts



class MaskedL1(torch.nn.Module):

    def forward(self, x, target, lens):
        target.requires_grad = False
        max_len = target.size(2)
        mask = pad_mask(lens, max_len)
        mask = mask.unsqueeze(1).expand_as(x)
        loss = F.l1_loss(
            x * mask, target * mask, reduction='sum')
        return loss / mask.sum()

# Adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def pad_mask(lens, max_len):
    batch_size = lens.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range = seq_range.unsqueeze(0)
    seq_range = seq_range.expand(batch_size, max_len)
    if lens.is_cuda:
        seq_range = seq_range.cuda()
    lens = lens.unsqueeze(1)
    lens = lens.expand_as(seq_range)
    mask = seq_range < lens
    return mask.float()

def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def train(run_id: str, model_type: str, syn_dir: str, models_dir: str, save_every: int, threads: int,
          backup_every: int, force_restart:bool, vis_every: int, visdom_server: str, no_visdom: bool):

    syn_dir = Path(syn_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)

    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)
    
    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")

    # Initialize Accelerator
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Accelerator process count: {0}".format(accelerator.num_processes))
        print("Checkpoint path: {}".format(weights_fpath))
        print("Loading training data from: {}".format(syn_dir))
        print("Using model: {}".format(model_type))
    
    # Book keeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # Let accelerator handle device
    device = accelerator.device

    # Init the model
    try:
        model = base.init_syn_model(model_type, device)
    except NotImplementedError as e:
        print(str(e))
        return

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())

    # Initialize the model if not initialized yet
    if force_restart or not weights_fpath.exists():
        accelerator.wait_for_everyone()
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                print("\nStarting the training of %s from scratch\n" % model_type)
                save(accelerator, model, weights_fpath)

                # Embeddings metadata
                char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
                with open(char_embedding_fpath, "w", encoding="utf-8") as f:
                    for symbol in symbols:
                        if symbol == " ":
                            symbol = "\\s"  # For visual purposes, swap space with \s

                        f.write("{}\n".format(symbol))

    # Model has been initialized - Load the weights
    print("{0} - Loading weights at {1}".format(device, weights_fpath))
    load(model, device, weights_fpath, optimizer)
    print("{0} - Model weights loaded from step {1}".format(device, model.get_step()))
    
    # Initialize the dataset
    dataset = SynthesizerDataset(syn_dir, base.get_model_train_elements(model_type))

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

    # Init epoch information
    epoch = 0
    max_step = 0

    # Determine which TTS schedule to use
    if model_type == base.MODEL_TYPE_TACOTRON:
        tts_schedule = hp_tacotron.tts_schedule
    elif model_type == base.MODEL_TYPE_FORWARD_TACOTRON:
        tts_schedule = hp_forward_tacotron.tts_schedule

    # Iterate over training schedule
    for i, session in enumerate(tts_schedule):
        # Update epoch information
        epoch += 1
        epoch_steps = max_step

        # Unwrap model after each epoch (if necessary) for re-calibration
        model = accelerator.unwrap_model(model)

        # Get the current step being processed
        current_step = model.get_step() + 1

        # Fetch params from session
        if model_type == base.MODEL_TYPE_TACOTRON:
            r, loops, batch_size, sgdr_init_lr, sgdr_final_lr = session
            # Update Model params
            model.r = r
        elif model_type == base.MODEL_TYPE_FORWARD_TACOTRON:
            loops, batch_size, sgdr_init_lr, sgdr_final_lr = session
            r = 1

        # Init dataloader
        data_loader = DataLoader(dataset,
                                 collate_fn=lambda batch: collate_synthesizer(batch, r),
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

        # Calc SGDR values
        sgdr_lr_stepping = (sgdr_init_lr - sgdr_final_lr) / np.ceil((total_samples * loops) / overall_batch_size).astype(np.int32)
        lr = sgdr_init_lr - (sgdr_lr_stepping * ((current_step-1) - epoch_steps))

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(hp_tacotron.tts_schedule) - 1:
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
            if model_type == base.MODEL_TYPE_TACOTRON:
                simple_table([("Epoch", epoch),
                          (f"Remaining Steps with r={r}", str(training_steps) + " Steps"),
                          ("Batch Size", batch_size),
                          ("Init LR", lr),
                          ("LR Stepping", sgdr_lr_stepping),
                          ("Outputs/Step (r)", r)])
            elif model_type == base.MODEL_TYPE_FORWARD_TACOTRON:
                simple_table([("Epoch", epoch),
                              (f"Remaining Steps", str(training_steps) + " Steps"),
                              ("Batch Size", batch_size),
                              ("Init LR", lr),
                              ("LR Stepping", sgdr_lr_stepping)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        # Training loop
        while current_step < max_step:
            #for step, (texts, mels, embeds, idx, mel_lens) in enumerate(data_loader, current_step):
            for step, (idx, utterance_ids, texts, text_lens, mels, mel_lens, embeds, durations, attentions, alignments, phoneme_pitchs, phoneme_energies ) in enumerate(data_loader, current_step):
                current_step = step
                start_time = time.time()
                model.train() # TODO: Verify this works as intended

                # Break out of loop to update training schedule
                if current_step > max_step:
                    break

                # Update lr
                lr = sgdr_init_lr - (sgdr_lr_stepping * ((current_step-1) - epoch_steps))
                for p in optimizer.param_groups:
                    p["lr"] = lr

                # Forward pass
                loss = None
                if model_type == base.MODEL_TYPE_TACOTRON:
                    # Generate stop tokens for training
                    stop = torch.ones(mels.shape[0], mels.shape[2])
                    for j, k in enumerate(idx):
                        stop[j, :int(dataset.metadata[k][2]) - 1] = 0
                    # Training step
                    loss, attention, m2_hat, texts, mels, embeds = tacotron_forward_pass(model, device, texts, mels, embeds, stop)
                elif model_type == base.MODEL_TYPE_FORWARD_TACOTRON:
                    # Training step
                    loss, mel_hat, mel_post, pitch_hat, energy_hat, texts, mels, embeds, durations, mel_lens, phoneme_pitchs, phoneme_energies = forward_tacotron_forward_pass(model, device, texts, text_lens, mels, embeds, durations, mel_lens, phoneme_pitchs, phoneme_energies)
                else:
                    raise NotImplementedError("Training not implemented for model of type '%s'. Aborting..." % model_type)

                # Backward pass
                optimizer.zero_grad()
                accelerator.backward(loss)

                if model_type == base.MODEL_TYPE_TACOTRON and hp_tacotron.tts_clip_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), hp_tacotron.tts_clip_grad_norm)
                elif model_type == base.MODEL_TYPE_FORWARD_TACOTRON and hp_forward_tacotron.clip_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), hp_forward_tacotron.clip_grad_norm)

                optimizer.step()

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                # Stream update on training progress
                if accelerator.is_local_main_process:
                    epoch_step = step - epoch_steps
                    epoch_max_step = max_step - epoch_steps
                    msg = f"| Epoch: {epoch} ({epoch_step}/{epoch_max_step}) | LR: {lr:#.6} | Loss: {loss_window.average:#.4} | {1./time_window.average:#.2} steps/s | Step: {step} | "
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
                    epoch_eval = hp_tacotron.tts_eval_interval == -1 and step == max_step  # If epoch is done
                    step_eval = hp_tacotron.tts_eval_interval > 0 and step % hp_tacotron.tts_eval_interval == 0  # Every N steps
                    if epoch_eval or step_eval:
                        for sample_idx in range(hp_tacotron.tts_eval_num_samples):
                            # At most, generate samples equal to number in the batch
                            if sample_idx + 1 <= len(texts):
                                if model_type == base.MODEL_TYPE_TACOTRON:
                                    # Remove padding from mels using frame length in metadata
                                    mel_length = int(dataset.metadata[idx[sample_idx]][2])
                                    mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                                    target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                                    attention_len = mel_length // r

                                    eval_tacotron_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                                        mel_prediction=mel_prediction,
                                                        target_spectrogram=target_spectrogram,
                                                        input_seq=np_now(texts[sample_idx]),
                                                        step=step,
                                                        plot_dir=plot_dir,
                                                        mel_output_dir=mel_output_dir,
                                                        wav_dir=wav_dir,
                                                        sample_num=sample_idx + 1,
                                                        loss=loss)

                                elif model_type == base.MODEL_TYPE_FORWARD_TACOTRON:
                                    mel_length = int(dataset.metadata[idx[sample_idx]][2])
                                    m1_hat = np_now(mel_hat[sample_idx])[:mel_length]
                                    m2_hat = np_now(mel_post[sample_idx])[:mel_length]
                                    m_target = np_now(mels[sample_idx])[:mel_length]

                                    input_seq = texts[sample_idx:(sample_idx + 1), :text_lens[sample_idx]]
                                    spk_emb = embeds[sample_idx:(sample_idx + 1)]
                                    pitch = np_now(phoneme_pitchs[sample_idx])
                                    pitch_gta = np_now(pitch_hat.squeeze()[sample_idx])
                                    energy = np_now(phoneme_energies[sample_idx])
                                    energy_gta = np_now(energy_hat.squeeze()[sample_idx])

                                    generate_plots(model,
                                                   plot_dir,
                                                   wav_dir,
                                                   step,
                                                   sample_idx + 1,
                                                   input_seq,
                                                   spk_emb,
                                                   m1_hat,
                                                   m2_hat,
                                                   m_target,
                                                   pitch,
                                                   pitch_gta,
                                                   energy,
                                                   energy_gta)

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
                save(accelerator, model, backup_fpath, optimizer)

def tacotron_forward_pass(model, device, texts, mels, embeds, stop):
    # Move data to device
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

    return loss, attention, m2_hat, texts, mels, embeds

def forward_tacotron_forward_pass(model, device, texts, text_lens, mels, embeds, durations, mel_lens, phoneme_pitchs, phoneme_energies):
    # Move data to device
    texts = texts.to(device)
    text_lens = text_lens.to(device)
    mels = mels.to(device)
    embeds = embeds.to(device)
    durations = durations.to(device)
    mel_lens = mel_lens.to(device)
    phoneme_pitchs = phoneme_pitchs.to(device)
    phoneme_energies = phoneme_energies.to(device)

    # Prepare Pitch & energy values
    pitch_zoneout_mask = torch.rand(texts.size()) > hp_forward_tacotron.pitch_zoneout
    energy_zoneout_mask = torch.rand(texts.size()) > hp_forward_tacotron.energy_zoneout
    pitch_target = phoneme_pitchs.detach().clone()
    energy_target = phoneme_energies.detach().clone()
    phoneme_pitchs = phoneme_pitchs * pitch_zoneout_mask.to(device).float()
    phoneme_energies = phoneme_energies * energy_zoneout_mask.to(device).float()

    # Forward Pass
    mel_hat, mel_post, dur_hat, pitch_hat, energy_hat = model(texts, mels, durations, embeds, mel_lens, phoneme_pitchs, phoneme_energies)

    # Calculate loss
    l1_loss = MaskedL1()

    m1_loss = l1_loss(mel_hat, mels, mel_lens)
    m2_loss = l1_loss(mel_post, mels, mel_lens)

    dur_loss = l1_loss(dur_hat.unsqueeze(1), durations.unsqueeze(1), text_lens)
    pitch_loss = l1_loss(pitch_hat, pitch_target.unsqueeze(1), text_lens)
    energy_loss = l1_loss(energy_hat, energy_target.unsqueeze(1), text_lens)

    loss = m1_loss + m2_loss \
           + hp_forward_tacotron.duration_loss_factor * dur_loss \
           + hp_forward_tacotron.pitch_loss_factor * pitch_loss \
           + hp_forward_tacotron.energy_loss_factor * energy_loss

    return loss, mel_hat, mel_post, pitch_hat, energy_hat, texts, mels, embeds, durations, mel_lens, phoneme_pitchs, phoneme_energies

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

def eval_tacotron_model(attention, mel_prediction, target_spectrogram, input_seq, step,
                        plot_dir, mel_output_dir, wav_dir, sample_num, loss):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # Save target wav for comparison
    target_wav = audio.inv_mel_spectrogram(target_spectrogram.T)
    target_wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}_target.wav".format(step, sample_num))
    audio.save_wav(target_wav, str(target_wav_fpath), sr=sp.sample_rate)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=sp.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // sp.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))

def generate_plots(model, plot_dir, wav_dir, step, sample_num, input_seq, spk_emb, m1_hat, m2_hat, m_target, pitch, pitch_gta, energy, energy_gta):
    # Plot all figures
    m1_hat_fig = plot_mel(m1_hat)
    m2_hat_fig = plot_mel(m2_hat)
    m_target_fig = plot_mel(m_target)
    pitch_fig = plot_pitch(pitch)
    pitch_gta_fig = plot_pitch(pitch_gta)
    energy_fig = plot_pitch(energy)
    energy_gta_fig = plot_pitch(energy_gta)

    # Save plots
    save_figure(m1_hat_fig, plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}_gta_linear".format(step, sample_num)))
    save_figure(m2_hat_fig, plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}_gta_postnet".format(step, sample_num)))
    save_figure(m_target_fig, plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}_target".format(step, sample_num)))
    save_figure(pitch_fig, plot_dir.joinpath("step-{}-pitch_sample_{}_target".format(step, sample_num)))
    save_figure(pitch_gta_fig, plot_dir.joinpath("step-{}-pitch_sample_{}_gta".format(step, sample_num)))
    save_figure(energy_fig, plot_dir.joinpath("step-{}-energy_sample_{}_target".format(step, sample_num)))
    save_figure(energy_gta_fig, plot_dir.joinpath("step-{}-energy_sample_{}_gta".format(step, sample_num)))

    # Save target wav for comparison
    target_wav = audio.inv_mel_spectrogram(m_target)
    target_wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}_target.wav".format(step, sample_num))
    audio.save_wav(target_wav, str(target_wav_fpath), sr=sp.sample_rate)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(m2_hat)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=sp.sample_rate)

    # Generate for speaker embedding and sequence - TODO: Not sure how this differs from the previous prediction actually
    mel, mel_post, dur_hat, pitch_hat, energy_hat = model.generate(input_seq, spk_emb)
    m1_hat = np_now(mel.squeeze())
    m2_hat = np_now(mel_post.squeeze())

    # Plot figures for generated
    m1_hat_fig = plot_mel(m1_hat)
    m2_hat_fig = plot_mel(m2_hat)
    pitch_gen_fig = plot_pitch(np_now(pitch_hat.squeeze()))
    energy_gen_fig = plot_pitch(np_now(energy_hat.squeeze()))

    # Save plots
    save_figure(m1_hat_fig, plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}_gen_linear".format(step, sample_num)))
    save_figure(m2_hat_fig, plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}_gen_postnet".format(step, sample_num)))
    save_figure(pitch_gen_fig, plot_dir.joinpath("step-{}-pitch_sample_{}_gen".format(step, sample_num)))
    save_figure(energy_gen_fig, plot_dir.joinpath("step-{}-energy_sample_{}_gen".format(step, sample_num)))

    # save griffin lim inverted generated wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(m2_hat)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}_generated.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=sp.sample_rate)




















