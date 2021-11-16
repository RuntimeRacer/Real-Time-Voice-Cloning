from typing import Dict
from multiprocess.pool import ThreadPool
from encoder.params_data import *
from encoder.config import librispeech_datasets, libritts_datasets, voxceleb_datasets, slr_datasets, other_datasets, anglophone_nationalites
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


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension, skip_existing, threads, logger):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)

        # Create an output directory with that name, as well as a txt file containing a
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")

        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")

        # get all the audio files
        source_files = list(speaker_dir.glob("**/*.%s" % extension))

        # cap at 75? 
        # FIXME: Add a feature to normalize amount of files per speaker, based on average amount of files per speaker in the dataset, rather than an arbitrary value.
        if len(source_files) > 75:
            source_files = source_files[:75]

        # Define npz output file and dict for the utterance data
        outpath = speaker_out_dir.joinpath("{0}_combined.npz".format(speaker_name))
        npz_data = {}
        try:
            # Try to load the existing combined file
            npz_data = np.load(outpath)
        except FileNotFoundError:
            print("No existing .npz file found for speaker {0}".format(speaker_name))
            pass

        # Iterate through the source files and add them to the NPZ file 
        for in_fpath in source_files:
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in npz_data:
                sources_file.write("%s,%s\n" % (out_fname, in_fpath))
                continue

            # Load and preprocess the waveform, discard those that are too short
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) < partials_n_frames:
                continue

            # Create the mel spectrogram
            # out_fpath = speaker_out_dir.joinpath(out_fname)
            # np.save(out_fpath, wav)
            npz_data[out_fname] = wav
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

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
def encoder_preprocess_dataset(datasets_root: Path, out_dir: Path, dataset_paths: set, file_type="wav", skip_existing=False, threads=8):
    for dataset_name in dataset_paths:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, file_type, skip_existing, threads, logger)


def preprocess_librispeech_other(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    for dataset_name in librispeech_datasets["train"]["other"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac", skip_existing, threads, logger)

def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    # Initialize the preprocessing
    for dataset_name in voxceleb_datasets["voxceleb1"]["train"]:
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Get the contents of the meta file
        # with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        #     metadata = [line.split("\t") for line in metafile][1:]
        
        # # Select the ID and the nationality, filter out non-anglophone speakers
        # nationalities = {line[0]: line[3] for line in metadata}
        # keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
        #                     nationality.lower() in anglophone_nationalites]
        # print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." % 
        #       (len(keep_speaker_ids), len(nationalities)))
        
        # # Get the speaker directories for anglophone speakers only
        speaker_dirs = dataset_root.glob("*")
        # speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
        #                 speaker_dir.name in keep_speaker_ids]
        # print("VoxCeleb1: found %d anglophone speakers on the disk, %d missing (this is normal)." % 
        #       (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))

        # Preprocess all speakers
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                skip_existing, threads, logger)

def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    # Initialize the preprocessing
    for dataset_name in voxceleb_datasets["voxceleb2"]["train"]:
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Get the speaker directories
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                skip_existing, threads, logger)

def preprocess_libritts_other(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    for dataset_name in libritts_datasets["train"]["other"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return 
        
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                 skip_existing, threads, logger)

def preprocess_vctk(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    # Initialize the preprocessing
    for dataset_name in other_datasets["VCTK"]:
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Get the speaker directories
        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))

        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                skip_existing, threads, logger)

def preprocess_slr(datasets_root: Path, out_dir: Path, slr_dataset=None, skip_existing=False, threads=8):
    for dataset_name in slr_datasets[slr_dataset]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

        # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                                 skip_existing, threads, logger)

def preprocess_commonvoice(datasets_root: Path, out_dir: Path, lang=None, skip_existing=False, threads=8):
    # simple dataset path
    dataset_name = "CommonVoice/{0}/speakers".format(lang)

    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Preprocess all speakers
    speaker_dirs = sorted(list(dataset_root.glob("*")))

    # speaker_dirs = speaker_dirs[0:4000] (complete)
    # speaker_dirs = speaker_dirs[4000:5000] (complete)
    # speaker_dirs = speaker_dirs[5000:7000] (complete)
    # speaker_dirs = speaker_dirs[7000:8000] (complete)
    # speaker_dirs = speaker_dirs[8000:9000] (in-progress)
    # speaker_dirs = speaker_dirs[9000:] (in-progress)

    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, threads, logger)

def preprocess_nasjonal(datasets_root: Path, out_dir: Path, lang=None, skip_existing=False, threads=8):
    # simple dataset path
    dataset_name = "nasjonal-bank/{0}/speakers".format(lang)

    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Preprocess all speakers
    speaker_dirs = sorted(list(dataset_root.glob("*")))

    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, threads, logger)

def preprocess_timit(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    # simple dataset path
    dataset_name = "TIMIT/speakers"

    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Preprocess all speakers
    speaker_dirs = sorted(list(dataset_root.glob("*")))

    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, threads, logger)

def preprocess_tedlium(datasets_root: Path, out_dir: Path, skip_existing=False, threads=8):
    # simple dataset path
    dataset_name = "TEDLIUM_release-3/data/speakers_verified"

    # Initialize the preprocessing
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Preprocess all speakers
    speaker_dirs = sorted(list(dataset_root.glob("*")))

    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, threads, logger)
