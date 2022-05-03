import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence
import json

from hparams.config import sp, preprocessing


class SynthesizerDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, pitch_dir: Path, embed_dir: Path):
        self.metadata_fpath = metadata_fpath
        print("Using inputs from:\n\t%s\n\t%s\n\t%s\n\t%s" % (self.metadata_fpath, mel_dir, pitch_dir, embed_dir))

        metadata = []
        with self.metadata_fpath.open("r") as metadata_file:
            metadata_dict = json.load(metadata_file)
            for speaker, lines in metadata_dict.items():
                metadata.extend([line.split("|") for line in lines])
        
        mel_fnames = [x[1] for x in metadata if int(x[5])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        pitch_fnames = [x[2] for x in metadata if int(x[5])]
        pitch_fpaths = [pitch_dir.joinpath(fname) for fname in pitch_fnames]
        embed_fnames = [x[3] for x in metadata if int(x[5])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        self.samples_fpaths = list(zip(mel_fpaths, pitch_fpaths, embed_fpaths))
        self.samples_texts = [x[6].strip() for x in metadata if int(x[5])]
        #self.samples_texts = [np.asarray(text_to_sequence(x[5].strip(), preprocessing.tts_cleaner_names)).astype(np.int32) for x in metadata if int(x[4])]
        self.metadata = metadata
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index
        if index is list:
            index = index[0]

        mel_path, pitch_path, embed_path = self.samples_fpaths[index]
        mel = np.load(mel_path).T.astype(np.float32)

        # Load the pitch
        pitch = np.load(pitch_path)
        
        # Load the embed
        embed = np.load(embed_path)

        # Get the text and clean it
        text = text_to_sequence(self.samples_texts[index], preprocessing.tts_cleaner_names)

        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        #return self.samples_texts[index], mel.astype(np.float32), embed.astype(np.float32), index
        return text, mel.astype(np.float32), pitch.astype(np.float32), embed.astype(np.float32), index

    def __len__(self):
        return len(self.samples_fpaths)

    def get_logs(self):
        speakers = 0
        utterances = 0

        log_string = ""
        with self.metadata_fpath.open("r") as metadata_file:
            metadata_dict = json.load(metadata_file)
            for speaker, lines in metadata_dict.items():
                speakers += 1
                utterances += len(lines)

        log_string += "Speakers: {0}\n".format(speakers)
        log_string += "Utterances: {0}\n".format(utterances)
        log_string += "Avg. Utterance / Speaker: {0}\n".format(utterances / speakers)
        return log_string


def collate_synthesizer(batch, r):
    # Text
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel spectrogram
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1 
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r 

    # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    if preprocessing.symmetric_mels:
        mel_pad_value = -1 * sp.max_abs_value
    else:
        mel_pad_value = 0

    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)

    # Audio Pitch
    pitch = np.array([x[2] for x in batch])

    # Speaker embedding (SV2TTS)
    embeds = np.array([x[3] for x in batch])

    # Index (for vocoder preprocessing)
    indices = [x[4] for x in batch]

    # Convert all to tensor
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    pitch = torch.tensor(pitch)
    embeds = torch.tensor(embeds)

    return chars, mel, pitch, embeds, indices

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)
