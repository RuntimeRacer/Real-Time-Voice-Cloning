from typing import Dict
from multiprocess.pool import ThreadPool
from encoder.params_data import *
from datetime import datetime
from encoder import audio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """
    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        from encoder import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()


def _init_preprocess_dataset(dataset_name, datasets_root, out_dir) -> (Path, DatasetLog):
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension, skip_existing, min, max, threads, logger):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)

        # get all the audio files
        source_files = list(speaker_dir.glob("**/*.%s" % extension))

        # Skip speakers with too few files
        if len(source_files) < min:
            print("Skipping speaker {0} due to too few recordings.".format(speaker_name))
            return

        # Create an output directory with that name, as well as a txt file containing a
        # reference to each source file. 
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")      
        # Gather all audio files for that speaker recursively.  
        sources_file = sources_fpath.open("a" if skip_existing else "w")

        # Limit amount of speaker files if too many
        if len(source_files) > max:
            random.shuffle(source_files)
            source_files = source_files[:max]

        # Define npz output file and dict for the utterance data
        outpath = speaker_out_dir.joinpath("combined.npz")
        npz_data = {}
        try:
            # Try to load the existing combined file
            npz_data = np.load(outpath)
        except FileNotFoundError:
            # print("No existing .npz file found for speaker {0}".format(speaker_name))
            pass

        # Iterate through the source files and add them to the NPZ file 
        for in_fpath in source_files:
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in npz_data:
                #sources_file.write("%s,%s\n" % (out_fname, in_fpath))
                continue

            # Load and preprocess the waveform, discard those that are too short
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue

            # Create the mel spectrogram, discard those that are too short
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            # out_fpath = speaker_out_dir.joinpath(out_fname)
            # np.save(out_fpath, wav)
            npz_data[out_fname] = frames
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

        # Ensure there aren't too many files in npz on second run
        kill_amount = len(npz_data) - max
        if kill_amount > 0:
            # Delete random file keys
            file_keys = list(npz_data.keys())
            random.shuffle(file_keys)
            while kill_amount > 0:
                for key in file_keys:
                    del npz_data[key]
                    kill_amount-=1
        
        # Write npz and sources file
        np.savez(outpath, **npz_data)
        sources_file.close()

    # Process the utterances for each speaker
    with ThreadPool(threads) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                  unit="speakers"))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)

# encoder_preprocess_dataset is intended to simplify the handling of dataset preprocessing. 
# Since we got pre-pre-processing scripts which do the heavy lifting towards getting all data into the expected format,
# this function works for anything that follows LibriSpeech / LibriTTS conventions or has been pre-pre-processed to match them.
def encoder_preprocess_dataset(
    datasets_root: Path, out_dir: Path, dataset_paths: set, file_type="wav", skip_existing=False, 
    min=12, max=40, threads=8
    ):
    for dataset_name in dataset_paths:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            # Skip this folder since it is missing, but try nex tone
            continue

        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, file_type, skip_existing, min, max, threads, logger)
