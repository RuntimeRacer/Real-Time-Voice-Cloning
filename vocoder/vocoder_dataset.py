import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hparams.config import wavernn as hp_wavernn, sp
from vocoder import audio


class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path):
        self.metadata_fpath = metadata_fpath
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (self.metadata_fpath, mel_dir, wav_dir))

        metadata = []
        with self.metadata_fpath.open("r") as metadata_file:
            metadata_dict = json.load(metadata_file)
            for line in metadata_dict.values():
                metadata.extend([line.split("|")])
        
        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [mel_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        self.samples_texts = [x[5].strip() for x in metadata if int(x[4])] if hp_wavernn.anomaly_detection else []
        self.metadata = metadata
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / sp.max_abs_value
        
        # Load the wav
        wav = np.load(wav_path)
        if sp.preemphasis:
            wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)
        
        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // sp.hop_size + 1) * sp.hop_size - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        assert len(wav) >= mel.shape[1] * sp.hop_size
        wav = wav[:mel.shape[1] * sp.hop_size]
        assert len(wav) % sp.hop_size == 0
        
        # Quantize the wav
        if hp_wavernn.mode == 'RAW':
            if hp_wavernn.mu_law:
                quant = audio.encode_mu_law(wav, mu=2 ** hp_wavernn.bits)
            else:
                quant = audio.float_2_label(wav, bits=hp_wavernn.bits)
        elif hp_wavernn.mode == 'MOL':
            quant = audio.float_2_label(wav, bits=16)
            
        return mel.astype(np.float32), quant.astype(np.int64), index

    def __len__(self):
        return len(self.samples_fpaths)

    def get_logs(self):
        samples = len(self.samples_fpaths)
        log_string = "Samples: {0}\n".format(samples)
        return log_string

        
def collate_vocoder(batch):
    if hp_wavernn.anomaly_detection:
        # collections of data for analyzing the batch
        src_mel_data = [x[0] for x in batch]
        src_wav_data = [x[1] for x in batch]
        indices = [x[2] for x in batch]
    src_data = (src_mel_data, src_wav_data, indices) if hp_wavernn.anomaly_detection else (None, None, None)

    # preprocessing for vocoder training
    mel_win = hp_wavernn.seq_len // sp.hop_size + 2 * hp_wavernn.pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp_wavernn.pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp_wavernn.pad) * sp.hop_size for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp_wavernn.seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp_wavernn.seq_len]
    y = labels[:, 1:]

    bits = 16 if hp_wavernn.mode == 'MOL' else hp_wavernn.bits

    x = audio.label_2_float(x.float(), bits)

    if hp_wavernn.mode == 'MOL' :
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels, src_data