import atexit
import json
import os
import time
from pathlib import Path

import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from hparams.config import tacotron as hp_tacotron, sp, preprocessing, sv2tts

from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import (SynthesizerDataset,
                                             collate_synthesizer)
from synthesizer.utils.symbols import symbols


def run_synthesis(in_dir, out_dir, model_dir, skip_existing, threads=2):
    # This generates ground truth-aligned mels for vocoder training
    synth_dir = Path(out_dir).joinpath("mels_gta")
    synth_dir.mkdir(exist_ok=True, parents=True)
    print(str(hp_tacotron))

    # Initialize Accelerator
    accelerator = Accelerator()

    # Let accelerator handle device
    device = accelerator.device
    print("Synthesizer using device:", device)

    # Instantiate Tacotron model
    model = Tacotron(embed_dims=hp_tacotron.embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hp_tacotron.encoder_dims,
                     decoder_dims=hp_tacotron.decoder_dims,
                     n_mels=sp.num_mels,
                     fft_bins=sp.num_mels,
                     postnet_dims=hp_tacotron.postnet_dims,
                     encoder_K=hp_tacotron.encoder_K,
                     lstm_dims=hp_tacotron.lstm_dims,
                     postnet_K=hp_tacotron.postnet_K,
                     num_highways=hp_tacotron.num_highways,
                     dropout=0., # Use zero dropout for gta mels
                     stop_threshold=hp_tacotron.stop_threshold,
                     speaker_embedding_size=sv2tts.speaker_embedding_size)

    # Load the weights
    model_dir = Path(model_dir)
    model_fpath = model_dir.joinpath(model_dir.stem).with_suffix(".pt")
    print("\nLoading weights at %s" % model_fpath)
    model.load(model_fpath)
    print("Tacotron weights loaded from step %d" % model.step)

    # Synthesize using same reduction factor as the model is currently trained
    r = np.int32(model.r)

    # Set model to eval mode (disable gradient and zoneout)
    model.eval()

    # Initialize the dataset
    in_dir = Path(in_dir)
    metadata_fpath = in_dir.joinpath("train.json")
    mel_dir = in_dir.joinpath("mels")
    embed_dir = in_dir.joinpath("embeds")

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir)
    data_loader = DataLoader(dataset,
                             collate_fn=lambda batch: collate_synthesizer(batch, r),
                             batch_size=preprocessing.synthesis_batch_size,
                             num_workers=threads,
                             shuffle=False,
                             pin_memory=True)

    # Accelerator code - optimize and prepare model
    model, data_loader = accelerator.prepare(model, data_loader)

    # Create synthesized dict and path of final file
    synthesized = {}
    synthesized_out_fpath = Path(out_dir).joinpath("synthesized.json".format(accelerator.state.process_index))

    # Read exsisting metadata in case existing data should be skipped
    if skip_existing and synthesized_out_fpath.is_file():
        with synthesized_out_fpath.open("r", encoding="utf-8") as synthesized_file:
            synthesized = json.load(synthesized_file)

    # Register a shutdown hook to safely store metadata process in case the process is interrupted
    atexit.register(save_synthesized_progress, synthesized, synthesized_out_fpath, out_dir, accelerator, True)

    # Generate GTA mels
    for i, (texts, mels, embeds, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):

        # Fixme this will regenerate also existing MELs, so currently we're just avoiding disk I/O after all
        _, mels_out, _, _ = model(texts, mels, embeds)

        for j, k in enumerate(idx):
            # Note: outputs mel-spectrogram files and target ones have same names, just different folders
            mel_filename = Path(synth_dir).joinpath(dataset.metadata[k][1])

            if skip_existing and str(mel_filename) in synthesized:
                continue

            mel_out = mels_out[j].detach().cpu().numpy().T

            # Use the length of the ground truth mel to remove padding from the generated mels
            mel_out = mel_out[:int(dataset.metadata[k][4])]

            # Write the spectrogram to disk
            np.save(mel_filename, mel_out, allow_pickle=False)

            # Write metadata into the synthesized file
            synthesized[str(mel_filename)] = "|".join(dataset.metadata[k])

    # Save metadata
    save_synthesized_progress(synthesized, synthesized_out_fpath, out_dir, accelerator, False)
    # Unregister shutdown hook
    atexit.unregister(save_synthesized_progress)

def save_synthesized_progress(synthesized_proc: dict, synthesized_out_fpath: Path, out_dir: str, accelerator: Accelerator, failure: bool):
    print("\nStoring results of different processing threads...")
    # Split output files during parallel processing
    synthesized_out_proc_fpath = Path(out_dir).joinpath("synthesized_{0}.json".format(accelerator.state.process_index))
    with synthesized_out_proc_fpath.open("w", encoding="utf-8") as synthesized_proc_file:
        json.dump(synthesized_proc, synthesized_proc_file)

    # FIXME: Sync does not work here properly when using CTRL+C. So instead wait a couple of seconds in case we're main process
    if not failure:
        accelerator.wait_for_everyone()

    with accelerator.local_main_process_first():
        if accelerator.is_local_main_process:
            time.sleep(5)

            print("\nCombining results of different processing threads...")
            # Read each dict and combine it into the final json
            synthesized = {}
            for proc_idx in range(accelerator.num_processes):
                # Split output files during parallel processing
                synthesized_out_proc_fpath = Path(out_dir).joinpath("synthesized_{0}.json".format(proc_idx))
                try:
                    with synthesized_out_proc_fpath.open("r", encoding="utf-8") as synthesized_file:
                        synthesized_sub_data = json.load(synthesized_file)
                        synthesized.update(synthesized_sub_data)
                except FileNotFoundError:
                    print("\nWARN: Unable to retrieve data for synthesized_{0}.json".format(proc_idx))
                    pass

            print("\nSaving synthesized metadata...")
            with synthesized_out_fpath.open("w", encoding="utf-8") as synthesized_file:
                json.dump(synthesized, synthesized_file)

            # Cleanup
            for proc_idx in range(accelerator.num_processes):
                # Split output files during parallel processing
                synthesized_out_proc_fpath = Path(out_dir).joinpath("synthesized_{0}.json".format(proc_idx))
                try:
                    os.remove(synthesized_out_proc_fpath)
                except FileNotFoundError:
                    pass

            print("Saved %d synthesized metadata entries to %s." % (len(synthesized), synthesized_file))
