from multiprocess.pool import ThreadPool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import config, inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
from audioread.exceptions import NoBackendError
from shutil import copyfile
import numpy as np
import librosa
import time
import atexit
import json


def save_metadata_progress(metadata: dict, metadata_fpath: Path):
    print("\nSaving training metadata...")
    with metadata_fpath.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file)
    print("Saved %d training metadata entries to %s." % (len(metadata), metadata_fpath))

def synthesizer_preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, hparams, no_alignments: bool, dataset_name: str, subfolders: str, audio_extensions: list, transcript_extension: str):
    # Gather the input directories
    dataset_root = datasets_root.joinpath(dataset_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

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
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, hparams=hparams, no_alignments=no_alignments, audio_extensions=audio_extensions, transcript_extension=transcript_extension)
    job = ThreadPool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, dataset_name, len(speaker_dirs), unit="speakers"):
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
    mel_frames = sum([int(m[4]) for m in metadata_lines])
    timesteps = sum([int(m[3]) for m in metadata_lines])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." % (len(metadata_lines), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata_lines))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata_lines))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata_lines))

def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool, audio_extensions: list, transcript_extension: str):
    speaker_metadata = {
        "speaker_dir": speaker_dir,
        "metadata": []
    }

    if no_alignments:
        # Gather the utterance audios and texts
        # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
        for extension in audio_extensions:
            wav_fpaths = speaker_dir.glob("**/*{0}".format(extension))

            for wav_fpath in wav_fpaths:
                utterance_id = "{0}_{1}".format(speaker_dir.name, wav_fpath.with_suffix("").name)

                # Get out paths
                mel_outpath = out_dir.joinpath("mels", "mel-%s.npy" % utterance_id)
                wav_outpath = out_dir.joinpath("audio", "audio-%s.npy" % utterance_id)

                # Load the audio waveform
                wav = None
                try:
                    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
                except (ValueError, RuntimeError, NoBackendError) as err:
                    # Unable to load.
                    print("Unable to load audio file {0}: {1}".format(wav_fpath, err))
                    continue

                if hparams.rescale:
                    wav = wav / np.abs(wav).max() * hparams.rescaling_max

                # Get the corresponding text
                # Check for .txt (for compatibility with other datasets)
                text_fpath = wav_fpath.with_suffix(transcript_extension)
                if not text_fpath.exists():
                    print("No transcript data for utterance {0} found: Missing file {1}. Skipping...\n".format(wav_fpath, text_fpath))
                    continue
                
                # Process text
                text = ""
                with text_fpath.open("r") as text_file:
                    text = text.join([line for line in text_file])
                    # Remove unwanted stuff from text
                    text = text.replace("\"", "")
                    text = text.replace("\\s", "")
                    text = text.replace("~", "")
                    text = text.strip()

                if len(text) == 0:
                    print("No transcript data for utterance {0} found: Missing file {1}. Skipping...\n".format(wav_fpath, text_fpath))
                    continue

                # Process the utterance
                output = process_utterance(wav, text, out_dir, mel_outpath, wav_outpath, utterance_id, hparams)
                if output is not None:
                    speaker_metadata["metadata"].append(output)

    else:
        # Process alignment file (LibriSpeech support)
        # Gather the utterance audios and texts
        # Gather the utterance audios and texts
        try:
            alignments_fpath = next(speaker_dir.glob("*.alignment.txt"))
            with alignments_fpath.open("r", encoding='utf-8') as alignments_file:
                alignments = [line.rstrip().split(" ") for line in alignments_file]
        except StopIteration as err:
            # A few alignment files will be missing
            print("Unable to process alignment file {0}: {1}".format(speaker_dir, err))
            return

        # Iterate over each entry in the alignments file
        for alignment in alignments:
            # expand the alignment
            # wav_fname, words, end_times
            wav_fname = alignment[0]
            words = alignment[1]
            end_times = alignment[2]
            transcript = " ".join(alignment[3:])

            # Find audio file for one of the allowed extension types
            wav_fpath = ""
            for extension in audio_extensions:
                wav_fpath = speaker_dir.joinpath(wav_fname + extension)
            
            if len(wav_fpath) == 0:
                print("Audio data for alignment {0} found: Missing file {1}. Skipping...\n" % alignment, wav_fname)
                continue

            words = words.replace("\"", "").split(",")
            end_times = list(map(float, end_times.replace("\"", "").split(",")))

            # Process each sub-utterance
            wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams, transcript)
            for i, (wav, text) in enumerate(zip(wavs, texts)):
                sub_basename = "%s_%02d" % (wav_fname, i)

                # Skip existing utterances if needed
                mel_outpath = out_dir.joinpath("mels", "mel-%s.npy" % sub_basename)
                wav_outpath = out_dir.joinpath("audio", "audio-%s.npy" % sub_basename)
                if skip_existing and mel_outpath.exists() and wav_outpath.exists():
                    continue

                # Process the utterance
                output = process_utterance(wav, text, out_dir, mel_outpath, wav_outpath, sub_basename, hparams)
                if output is not None:
                    speaker_metadata["metadata"].append(output)

    return speaker_metadata


def split_on_silences(wav_fpath, words, end_times, hparams, transcript):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

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
    mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < hparams.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
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
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
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


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, mel_fpath: Path, wav_fpath: Path, basename: str, hparams):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Trim silence
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
    
    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        #print("Skipped utterance {0} because it's too short".format(basename))
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        #print("Skipped utterance {0} because it's too long.".format(basename))
        return None

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = ThreadPool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
    
