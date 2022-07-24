from multiprocess.pool import ThreadPool, Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from synthesizer import batched as synthesizer_model
from synthesizer.models import base
from synthesizer.synthesizer_dataset import (SynthesizerDataset,
                                             collate_synthesizer)
from synthesizer.utils.duration_extractor import DurationExtractor
from synthesizer.utils.text import text_to_sequence
from synthesizer.utils.text import sequence_to_text_ascii

from accelerate import Accelerator
from pathlib import Path
from utils import logmmse
from utils.display import *
from tqdm import tqdm
from audioread.exceptions import NoBackendError
from shutil import copyfile
from config.hparams import sp, preprocessing
from config.paths import synthesizer

from torch.utils.data import DataLoader
import torch
import numpy as np
import pyworld as pw
import librosa
import time
import atexit
import json


def save_metadata_progress(metadata: dict, metadata_fpath: Path):
    print("\nSaving training metadata...")
    with metadata_fpath.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file)
    print("Saved %d training metadata entries to %s." % (len(metadata), metadata_fpath))


def synthesizer_preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, dataset_name: str, subfolders: str, audio_extensions: list, transcript_extension: str):
    # Gather the input directories
    dataset_root = datasets_root.joinpath(dataset_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath(synthesizer.mel_dir).mkdir(exist_ok=True)
    out_dir.joinpath(synthesizer.wav_dir).mkdir(exist_ok=True)

    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.json")
    metadata_bup_fpath = out_dir.joinpath("train_backup_{0}.json".format(time.time()))
    if metadata_fpath.is_file() and not skip_existing:
        # Backup so we don't accidentially delete sth.
        copyfile(metadata_fpath, metadata_bup_fpath)

    # Create metadata dict
    metadata = {}

    # Read exsisting metadata in case existing data should be skipped
    if skip_existing and metadata_fpath.is_file():
        with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    if skip_existing:
        speaker_dirs[:] = (speaker_dir for speaker_dir in speaker_dirs if not str(speaker_dir) in metadata)

    # Register a shutdown hook to safely store metadata process in case the process is interrupted
    atexit.register(save_metadata_progress, metadata, metadata_fpath)

    # Preprocess the dataset
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, audio_extensions=audio_extensions, transcript_extension=transcript_extension)
    job = ThreadPool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, dataset_name, len(speaker_dirs), unit="speakers", miniters=1):
        speaker_dir = str(speaker_metadata["speaker_dir"])
        metadata[speaker_dir] = []
        for metadatum in speaker_metadata["metadata"]:
            metadata[speaker_dir].append("|".join(str(x) for x in metadatum))

    # Save metadata file
    save_metadata_progress(metadata, metadata_fpath)
    # Unregister shutdown hook
    atexit.unregister(save_metadata_progress)

    # Verify the contents of the metadata file
    metadata_lines = []
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
        for speaker, utterances in metadata.items():
            metadata_lines.extend([line.split("|") for line in utterances])
    mel_frames = sum([int(m[2]) for m in metadata_lines])
    timesteps = sum([int(m[1]) for m in metadata_lines])
    sample_rate = sp.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." % (len(metadata_lines), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[3]) for m in metadata_lines))
    print("Max mel frames length: %d" % max(int(m[2]) for m in metadata_lines))
    print("Max audio timesteps length: %d" % max(int(m[1]) for m in metadata_lines))


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, audio_extensions: list, transcript_extension: str):
    speaker_metadata = {
        "speaker_dir": speaker_dir,
        "metadata": []
    }

    # Gather the utterance audios and texts
    # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
    for extension in audio_extensions:
        wav_fpaths = speaker_dir.glob("**/*{0}".format(extension))

        for wav_fpath in wav_fpaths:
            utterance_id = "{0}_{1}".format(speaker_dir.name, wav_fpath.with_suffix("").name)

            # Load the audio waveform
            wav = None
            try:
                wav, _ = librosa.load(path=str(wav_fpath), sr=sp.sample_rate)
            except (ValueError, RuntimeError, NoBackendError) as err:
                # Unable to load.
                tqdm.write("Unable to load audio file {0}: {1}".format(wav_fpath, err))
                continue

            if preprocessing.rescale:
                wav = wav / np.abs(wav).max() * preprocessing.rescaling_max

            # Get the corresponding text
            # Check for .txt (for compatibility with other datasets)
            text_fpath = wav_fpath.with_suffix(transcript_extension)
            if not text_fpath.exists():
                tqdm.write("No transcript data for utterance {0} found: Missing file {1}. Skipping...\n".format(wav_fpath, text_fpath))
                continue

            # Process text
            text = ""
            with text_fpath.open("r") as text_file:
                text = text.join([line for line in text_file])

            if len(text) == 0:
                tqdm.write("No transcript data for utterance {0} found: Missing file {1}. Skipping...\n".format(wav_fpath, text_fpath))
                continue

            # Process the utterance
            # Output: (utterance_id, len(wav), mel_frames, text)
            output = process_utterance(utterance_id, wav, text, out_dir)
            if output is not None:
                speaker_metadata["metadata"].append(output)

    return speaker_metadata


def split_on_silences(wav_fpath, words, end_times, transcript):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), sp.sample_rate)
    if preprocessing.rescale:
        wav = wav / np.abs(wav).max() * preprocessing.rescaling_max

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)

    # if the front and end are not blank, just skip it?
    if words[0] != "" and words[-1] != "":
        if transcript is not None:
            return [wav], [transcript]
        else:
            return [wav], [" ".join(words).replace("  ", " ")]

    # assert words[0] == "" and words[-1] == ""

    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= preprocessing.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * sp.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > sp.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, sp.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < preprocessing.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > sp.hop_size * preprocessing.max_mel_frames / sp.sample_rate:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * sp.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

    # # DEBUG: play the audio segments (run with -n=1)
    # import sounddevice as sd
    # if len(wavs) > 1:
    #     print("This sentence was split in %d segments:" % len(wavs))
    # else:
    #     print("There are no silences long enough for this sentence to be split:")
    # for wav, text in zip(wavs, texts):
    #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
    #     # when playing them. You shouldn't need to do that in your parsers.
    #     wav = np.concatenate((wav, [0] * 16000))
    #     print("\t%s" % text)
    #     sd.play(wav, 16000, blocking=True)
    # print("")

    return wavs, texts


def process_utterance(utterance_id: str, wav: np.ndarray, text: str, out_dir: Path):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - The audios, pitch data and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Trim silence over whole audio
    if preprocessing.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)

    # Trim start and end silence
    if preprocessing.trim_start_end_silence:
        wav = encoder.audio.trim_silence(wav, preprocessing.trim_silence_top_db)
    
    # Skip utterances that are too short
    if len(wav) < preprocessing.utterance_min_duration * sp.sample_rate:
        #print("Skipped utterance {0} because it's too short".format(basename))
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > preprocessing.max_mel_frames and preprocessing.clip_mels_length:
        #print("Skipped utterance {0} because it's too long.".format(basename))
        return None

    # Get out paths
    mel_outpath = out_dir.joinpath(synthesizer.mel_dir, "mel-%s.npy" % utterance_id)
    wav_outpath = out_dir.joinpath(synthesizer.wav_dir, "audio-%s.npy" % utterance_id)

    # Write the spectrogram and audio to disk
    np.save(mel_outpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_outpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return utterance_id, len(wav), mel_frames, text


def embed_utterance(utterance_id, synthesizer_root, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath = synthesizer_root.joinpath(synthesizer.wav_dir, "audio-%s.npy" % utterance_id)
    embed_fpath = synthesizer_root.joinpath(synthesizer.embed_dir, "embed-%s.npy" % utterance_id)

    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)

def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, skip_existing: bool, n_processes: int):
    wav_dir = synthesizer_root.joinpath(synthesizer.wav_dir)
    metadata_fpath = synthesizer_root.joinpath("train.json")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath(synthesizer.embed_dir)
    embed_dir.mkdir(exist_ok=True)

    # Gather the utterance IDs to derive file names from
    utterance_ids = []
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata_dict = json.load(metadata_file)
        for speaker, lines in metadata_dict.items():
            metadata = [line.split("|") for line in lines]
            utterance_ids.extend([(m[0]) for m in metadata])

    # Check for existing files
    if skip_existing:
        embedding_files = list(embed_dir.glob("embed-*.npy"))
        print(embedding_files[0])
        utterance_ids[:] = (utterance_id for utterance_id in utterance_ids if not str(embed_dir.joinpath("embed-%s.npy" % utterance_id)) in embedding_files)

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, synthesizer_root=synthesizer_root, encoder_model_fpath=encoder_model_fpath)
    job = ThreadPool(n_processes).imap(func, utterance_ids)
    list(tqdm(job, "Embedding", len(utterance_ids), unit="utterances", miniters=1))

def create_alignments(utterance, accelerator: Accelerator, synthesizer_root: Path, synthesizer_model_fpath: Path):
    if not synthesizer_model.is_loaded():
        synthesizer_model.load_tacotron_model(synthesizer_model_fpath, device=accelerator.device, use_tqdm=True)

    # Get text, audio wav, mel and embedding
    utterance_id, text = utterance
    wav_fpath = synthesizer_root.joinpath(synthesizer.wav_dir, "audio-%s.npy" % utterance_id)
    mel_fpath = synthesizer_root.joinpath(synthesizer.mel_dir, "mel-%s.npy" % utterance_id)
    embed_fpath = synthesizer_root.joinpath(synthesizer.embed_dir, "embed-%s.npy" % utterance_id)
    wav = np.load(wav_fpath)
    mel = np.load(mel_fpath).T.astype(np.float32)
    embed = np.load(embed_fpath)

    # Get the text and clean it
    text = text_to_sequence(text, preprocessing.cleaner_names)
    # Convert the list returned by text_to_sequence to a numpy array
    text = np.asarray(text).astype(np.int32)
    text = pad1d(text, len(text))

    # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    if preprocessing.symmetric_mels:
        mel_pad_value = -1 * sp.max_abs_value
    else:
        mel_pad_value = 0

    mel_len = mel.shape[-1]
    max_spec_len = mel_len + 1
    mel = pad2d(mel, max_spec_len, pad_value=mel_pad_value)

    # convert to tensors and add to device
    text = torch.tensor(np.stack([text])).long()
    mel = torch.tensor(np.stack([mel]))
    embed = torch.tensor(np.stack([embed]))

    # Use tacotron to generate attention
    att_batch = synthesizer_model.get_attention_batch(text, mel, embed)

    # Get alignment score
    mel_len = torch.tensor(np.stack([mel_len])).cpu()
    align_score_seq, _ = get_attention_score(att_batch, mel_len)
    align_score = float(align_score_seq[0])

    # Init the duration extractor
    duration_extractor = DurationExtractor(silence_threshold=preprocessing.silence_threshold,
                                           silence_prob_shift=preprocessing.silence_prob_shift)
    # Get raw pitch
    pitch, _ = pw.dio(wav.astype(np.float64), sp.sample_rate, frame_period=sp.hop_size / sp.sample_rate * 1000)
    pitch = pitch.astype(np.float32)

    # we use the standard alignment score and the more accurate attention score from the duration extractor
    text = text[0].cpu()
    mel_len = mel_len[0].cpu()
    mel = mel[0, :, :mel_len].cpu()
    att = att_batch[0, :mel_len, :].cpu()
    duration, att_score = duration_extractor(x=text, mel=mel, att=att)
    duration = np_now(duration).astype(np.int)

    if np.sum(duration) != mel_len:
        tqdm.write(f'WARNING: Sum of durations did not match mel length for item {utterance_id}!')

    # Get mel energy
    energy = np.linalg.norm(np.exp(mel), axis=0, ord=2)
    assert np.sum(duration) == mel_len

    # TODO: Not sure what exactly this does...
    durs_cum = np.cumsum(np.pad(duration, (1, 0)))
    pitch_char = np.zeros((duration.shape[0],), dtype=np.float32)
    energy_char = np.zeros((duration.shape[0],), dtype=np.float32)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        values = values[np.where(values < preprocessing.pitch_max_freq)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
        energy_values = energy[a:b]
        energy_char[idx] = np.mean(energy_values) if len(energy_values) > 0 else 0.0

    # Save items
    # FIXME: This is gonna be super slow and IO intensive. OK for evaluation but needs refactoring later.
    durations_dir = synthesizer_root.joinpath(synthesizer.duration_dir)
    attention_dir = synthesizer_root.joinpath(synthesizer.attention_dir)
    alignment_dir = synthesizer_root.joinpath(synthesizer.alignment_dir)
    phoneme_pitch_dir = synthesizer_root.joinpath(synthesizer.phoneme_pitch_dir)
    phoneme_energy_dir = synthesizer_root.joinpath(synthesizer.phoneme_energy_dir)
    np.save(str(durations_dir / f'duration-{utterance_id}.npy'), duration, allow_pickle=False)
    np.save(str(attention_dir / f'attention-{utterance_id}.npy'), att_score, allow_pickle=False)
    np.save(str(alignment_dir / f'alignment-{utterance_id}.npy'), align_score, allow_pickle=False)
    np.save(str(phoneme_pitch_dir / f'phoneme-pitch-{utterance_id}.npy'), pitch_char,
            allow_pickle=False)  # TODO: train_tacotron.py:73 (in FW-Taco repo): Check if this normalization is needed for anything
    np.save(str(phoneme_energy_dir / f'phoneme-energy-{utterance_id}.npy'), energy_char, allow_pickle=False)

def create_align_features(synthesizer_root: Path, synthesizer_model_fpath: Path, skip_existing: bool, n_processes: int):
    # Initialize the dataset
    mel_dir = synthesizer_root.joinpath(synthesizer.mel_dir)
    embed_dir = synthesizer_root.joinpath(synthesizer.embed_dir)
    metadata_fpath = synthesizer_root.joinpath("train.json")
    assert mel_dir.exists() and embed_dir.exists() and metadata_fpath.exists()

    # Create paths needed
    durations_dir = synthesizer_root.joinpath(synthesizer.duration_dir)
    attention_dir = synthesizer_root.joinpath(synthesizer.attention_dir)
    alignment_dir = synthesizer_root.joinpath(synthesizer.alignment_dir)
    phoneme_pitch_dir = synthesizer_root.joinpath(synthesizer.phoneme_pitch_dir)
    phoneme_energy_dir = synthesizer_root.joinpath(synthesizer.phoneme_energy_dir)
    durations_dir.mkdir(exist_ok=True)
    attention_dir.mkdir(exist_ok=True)
    alignment_dir.mkdir(exist_ok=True)
    phoneme_pitch_dir.mkdir(exist_ok=True)
    phoneme_energy_dir.mkdir(exist_ok=True)

    # Gather the utterance IDs to derive file names from
    utterances = []
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata_dict = json.load(metadata_file)
        for speaker, lines in metadata_dict.items():
            metadata = [line.split("|") for line in lines]
            utterances.extend([(m[0],m[3].strip()) for m in metadata if int(m[2])])

    # Init Accelerator
    torch.multiprocessing.set_start_method('spawn', force=True)
    accelerator = Accelerator()

    # Split dataset for the current process
    len_utterances = len(utterances)
    split_idx = int(len_utterances / accelerator.state.num_processes)
    acc_proc_id = accelerator.state.process_index
    if acc_proc_id == (accelerator.num_processes - 1):
        utterances = utterances[split_idx*acc_proc_id:]
    else:
        utterances = utterances[split_idx*acc_proc_id:split_idx*(acc_proc_id+1)]

    # Create alignments for the utterances in separate threads
    # for utterance in utterances:
    #     create_alignments(utterance, synthesizer_root=synthesizer_root, synthesizer_model_fpath=synthesizer_model_fpath)
    func = partial(create_alignments, accelerator=accelerator, synthesizer_root=synthesizer_root, synthesizer_model_fpath=synthesizer_model_fpath)
    job = torch.multiprocessing.Pool(n_processes).imap(func, utterances)
    list(tqdm(job, "Alignments", len(utterances), unit="utterances", miniters=1))

def get_attention_score(att, mel_lens, r=1):
    """
    Returns a tuple of scores (loc_score, sharp_score), where loc_score measures monotonicity and
    sharp_score measures the sharpness of attention peaks
    """

    with torch.no_grad():
        device = att.device
        mel_lens = mel_lens.to(device)
        b, t_max, c_max = att.size()

        # create mel padding mask
        mel_range = torch.arange(0, t_max, device=device)
        mel_lens = mel_lens // r
        mask = (mel_range[None, :] < mel_lens[:, None]).float()

        # score for how adjacent the attention loc is
        max_loc = torch.argmax(att, dim=2)
        max_loc_diff = torch.abs(max_loc[:, 1:] - max_loc[:, :-1])
        loc_score = (max_loc_diff >= 0) * (max_loc_diff <= r)
        loc_score = torch.sum(loc_score * mask[:, 1:], dim=1)
        loc_score = loc_score / (mel_lens - 1)

        # score for attention sharpness
        sharp_score, inds = att.max(dim=2)
        sharp_score = torch.mean(sharp_score * mask, dim=1)

        return loc_score, sharp_score

def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)