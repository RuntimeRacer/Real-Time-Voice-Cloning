import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config.hparams import sp
from vocoder import audio


class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path, vocoder_hparams):
        self.metadata_fpath = metadata_fpath
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (self.metadata_fpath, mel_dir, wav_dir))

        metadata = []
        with self.metadata_fpath.open("r") as metadata_file:
            metadata_dict = json.load(metadata_file)
            for line in metadata_dict.values():
                metadata.extend([line.split("|")])

        gta_fnames = [x[0] for x in metadata if int(x[2])]
        gta_fpaths = [mel_dir.joinpath("%s.npy" % fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[2])]
        wav_fpaths = [wav_dir.joinpath("audio-%s.npy" % fname) for fname in wav_fnames]
        self.vocoder_hparams = vocoder_hparams
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        self.metadata = metadata

        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):
        # Get paths from samples list
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / sp.max_abs_value
        # Load the wav
        wav = np.load(wav_path)

        # Quantize only relevant for wavernn
        if hasattr(self.vocoder_hparams, "mode"):
            if sp.preemphasis:
                wav = audio.pre_emphasis(wav)
            wav = np.clip(wav, -1, 1)

            # Fix for missing padding
            r_pad = (len(wav) // sp.hop_size + 1) * sp.hop_size - len(wav)
            wav = np.pad(wav, (0, r_pad), mode='constant')
            assert len(wav) >= mel.shape[1] * sp.hop_size
            wav = wav[:mel.shape[1] * sp.hop_size]
            assert len(wav) % sp.hop_size == 0

            # Quantize the wav TODO: This can be optimized to be done during preprocessing
            if self.vocoder_hparams.mode == 'MOL':
                quant = audio.float_2_label(wav, bits=16)
            else:
                if self.vocoder_hparams.mu_law:
                    quant = audio.encode_mu_law(wav, mu=2 ** self.vocoder_hparams.bits)
                else:
                    quant = audio.float_2_label(wav, bits=self.vocoder_hparams.bits)
            quant = quant.astype(np.int64)

            # WaveRNN Result
            return mel.astype(np.float32), quant, index
        else:
            # MelGAN result
            return wav.astype(np.float32), mel.T.astype(np.float32)

    def __len__(self):
        return len(self.samples_fpaths)

    def get_len(self):
        return len(self.samples_fpaths)

    def get_logs(self):
        samples = len(self.samples_fpaths)
        log_string = "Samples: {0}\n".format(samples)
        return log_string

        
def collate_wavernn(batch, vocoder_hparams):
    # Indices; used for debugging
    indices = [x[2] for x in batch]

    # preprocessing for vocoder training
    mel_win = vocoder_hparams.seq_len // sp.hop_size + 2 * vocoder_hparams.pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * vocoder_hparams.pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + vocoder_hparams.pad) * sp.hop_size for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + vocoder_hparams.seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :vocoder_hparams.seq_len]
    y = labels[:, 1:]

    bits = 16 if vocoder_hparams.mode == 'MOL' else vocoder_hparams.bits

    x = audio.label_2_float(x.float(), bits)

    if vocoder_hparams.mode == 'MOL':
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels, indices


def collate_melgan(batch, vocoder_hparams):
    # Set values used by melgan preprocessing
    batch_max_steps = vocoder_hparams.seq_len
    batch_max_frames = batch_max_steps // sp.hop_size
    start_offset = vocoder_hparams.aux_context_window
    end_offset = -(batch_max_frames + vocoder_hparams.aux_context_window)
    mel_threshold = batch_max_steps // sp.hop_size + 2 * vocoder_hparams.aux_context_window  # mel_win

    # check length
    batch = [
        _adjust_length(*b) for b in batch if len(b[1]) > mel_threshold
    ]
    xs, cs = [b[0] for b in batch], [b[1] for b in batch]

    # make batch with random cut
    c_lengths = [len(c) for c in cs]
    start_frames = np.array(
        [
            np.random.randint(start_offset, cl + end_offset)
            for cl in c_lengths
        ]
    )
    x_starts = start_frames * sp.hop_size
    x_ends = x_starts + batch_max_steps
    c_starts = start_frames - vocoder_hparams.aux_context_window
    c_ends = start_frames + batch_max_frames + vocoder_hparams.aux_context_window
    y_batch = [x[start:end] for x, start, end in zip(xs, x_starts, x_ends)]
    c_batch = [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)]

    # convert each batch to tensor, asuume that each item in batch has the same length
    y_batch, c_batch = np.array(y_batch), np.array(c_batch)
    y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
    c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

    # make input noise signal batch tensor if ParallelWaveGAN
    if vocoder_hparams.generator_type == "ParallelWaveGANGenerator":
        z_batch = torch.randn(y_batch.size())  # (B, 1, T)
        return (z_batch, c_batch), y_batch
    else:
        # Otherwise, return default batch
        return (c_batch,), y_batch


def _adjust_length(x, c):
    """Adjust the audio and feature lengths.

    Note:
        Basically we assume that the length of x and c are adjusted
        through preprocessing stage, but if we use other library processed
        features, this process will be needed.

    """
    if len(x) < len(c) * sp.hop_size:
        x = np.pad(x, (0, len(c) * sp.hop_size - len(x)), mode="edge")

    # check the legnth is valid
    assert len(x) == len(c) * sp.hop_size

    return x, c
