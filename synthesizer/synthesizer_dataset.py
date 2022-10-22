import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence, text_to_sequence_ascii
import json

from config.hparams import sp, preprocessing
from config.paths import synthesizer


class SynthesizerDataset(Dataset):
    def __init__(self, synthesizer_root: Path, elements_to_provide: list):
        self.synthesizer_root = synthesizer_root
        self.elements_to_provide = elements_to_provide

        # Get metadata file
        self.metadata_fpath = synthesizer_root.joinpath("train.json")
        assert self.metadata_fpath.exists()
        print("Using inputs from:\n\t%s" % (self.metadata_fpath))

        metadata = []
        with self.metadata_fpath.open("r") as metadata_file:
            metadata_dict = json.load(metadata_file)
            for speaker, lines in metadata_dict.items():
                metadata.extend([line.split("|") for line in lines])

        self.samples_fnames = [x[0] for x in metadata if int(x[2])]
        self.samples_texts = [x[3].strip() for x in metadata if int(x[2])]
        self.metadata = metadata
        
        print("Found %d samples" % len(self.samples_fnames))
    
    def __getitem__(self, index):  
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index
        if index is list:
            index = index[0]

        # Get utterance ID
        utterance_id = self.samples_fnames[index]
        # Get the text and clean it
        text = text_to_sequence(self.samples_texts[index], preprocessing.cleaner_names)
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        # initialize possible return values as empty np arrays
        mel = np.zeros(1)
        embed = np.zeros(1)
        duration = np.zeros(1)
        attention = np.zeros(1)
        alignment = np.zeros(1)
        phoneme_pitch = np.zeros(1)
        phoneme_energy = np.zeros(1)

        # Fill all the values if they should be provided.
        # Mel Spectogram
        if "mel" in self.elements_to_provide:
            mel_path = self.synthesizer_root.joinpath(synthesizer.mel_dir, "mel-%s.npy" % utterance_id)
            mel = np.load(mel_path).T.astype(np.float32)
        # Embedding
        if "embed" in self.elements_to_provide:
            embed_path = self.synthesizer_root.joinpath(synthesizer.embed_dir, "embed-%s.npy" % utterance_id)
            embed = np.load(embed_path)
        # Duration
        if "duration" in self.elements_to_provide:
            duration_path = self.synthesizer_root.joinpath(synthesizer.duration_dir, "duration-%s.npy" % utterance_id)
            duration = np.load(duration_path)
        # Attention
        if "attention" in self.elements_to_provide:
            attention_path = self.synthesizer_root.joinpath(synthesizer.attention_dir, "attention-%s.npy" % utterance_id)
            attention = np.load(attention_path)
        # Alignment
        if "alignment" in self.elements_to_provide:
            alignment_path = self.synthesizer_root.joinpath(synthesizer.alignment_dir, "alignment-%s.npy" % utterance_id)
            alignment = np.load(alignment_path)
        # Phoneme Pitch
        if "phoneme_pitch" in self.elements_to_provide:
            phoneme_pitch_path = self.synthesizer_root.joinpath(synthesizer.phoneme_pitch_dir, "phoneme-pitch-%s.npy" % utterance_id)
            phoneme_pitch = np.load(phoneme_pitch_path)
        # Phoneme Energy
        if "phoneme_energy" in self.elements_to_provide:
            phoneme_energy_path = self.synthesizer_root.joinpath(synthesizer.phoneme_energy_dir, "phoneme-energy-%s.npy" % utterance_id)
            phoneme_energy = np.load(phoneme_energy_path)

        return index, \
               text, \
               mel.astype(np.float32), \
               embed.astype(np.float32), \
               duration.astype(np.int32), \
               attention.astype(np.float32), \
               alignment.astype(np.float32), \
               phoneme_pitch.astype(np.float32), \
               phoneme_energy.astype(np.float32)

    def __len__(self):
        return len(self.samples_fnames)

    def get_len(self):
        return len(self.samples_fnames)

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

    # Index (for vocoder preprocessing)
    indices = [x[0] for x in batch]
    
    # Text
    x_lens = [len(x[1]) for x in batch]
    max_x_len = max(x_lens)
    x_lens = np.stack(x_lens)

    chars = [pad1d(x[1], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel spectrogram
    spec_lens = [x[2].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1 
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r 

    # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    if preprocessing.symmetric_mels:
        mel_pad_value = -1 * sp.max_abs_value
    else:
        mel_pad_value = 0

    mel = [pad2d(x[2], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)

    # Speaker embedding (SV2TTS)
    embeds = np.array([x[3] for x in batch])
    # Durations
    duration_lens = [len(x[4]) for x in batch]
    max_duration_len = max(duration_lens)
    durations = [pad1d(x[4], max_duration_len) for x in batch]
    durations = np.stack(durations)
    # Attentions
    attentions = np.array([x[5] for x in batch])
    # Alignments
    alignments = np.array([x[6] for x in batch])
    # Phoneme Pitch
    pitch_lens = [len(x[7]) for x in batch]
    max_pitch_len = max(pitch_lens)
    phoneme_pitch = [pad1d(x[7], max_pitch_len) for x in batch]
    phoneme_pitch = np.stack(phoneme_pitch)
    # Phoneme Energy
    energy_lens = [len(x[8]) for x in batch]
    max_energy_len = max(energy_lens)
    phoneme_energy = [pad1d(x[8], max_energy_len) for x in batch]
    phoneme_energy = np.stack(phoneme_energy)

    # Convert all to tensor
    chars = torch.tensor(chars).long()
    x_lens = torch.tensor(x_lens).long()
    mel = torch.tensor(mel)    
    spec_lens = torch.tensor(spec_lens)
    embeds = torch.tensor(embeds)
    durations = torch.tensor(durations)
    attentions = torch.tensor(attentions)
    alignments = torch.tensor(alignments)
    phoneme_pitch = torch.tensor(phoneme_pitch)
    phoneme_energy = torch.tensor(phoneme_energy)

    return indices, chars, x_lens, mel, spec_lens, embeds, durations, attentions, alignments, phoneme_pitch, phoneme_energy

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)
