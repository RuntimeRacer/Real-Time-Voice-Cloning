import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import argparse
import random
from multiprocess.pool import ThreadPool
from shutil import copyfile

# Functions
# - none

# Parser for Arguments
parser = argparse.ArgumentParser(description='Process common voice dataset for a language.')
parser.add_argument("datasets_root", type=Path, help=\
    "Path to the directory containing your CommonVoice datasets.")
parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument('--min', type=int, default=5, help=\
    'Minimum number of files per speaker')
parser.add_argument('--max', type=int, default=40, help=\
    'Maximum number of files per speaker')
parser.add_argument("-ft", "--fetch_transcripts", type=bool, default=False, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# Stats
speaker_count = 0

# Speaker files dict
speaker_hash = {}

base_dir = args.datasets_root
_, speaker_dirs, _ = next(os.walk(base_dir))

speaker_count = len(speaker_dirs)
print("Found {0} speakers.".format(speaker_count))

# sort the speaker_id/client_id by
sorted_speakers = sorted(speaker_dirs)

out_dir = base_dir
if out_dir != None:
    out_dir = args.out_dir

# if we have a speakers directory, remove it!
#if out_dir.joinpath("speakers").is_dir() == True:
#    rmtree(out_dir.joinpath("speakers"))

def process_speaker(speaker):
    # Get list of files in the folder
    # Only mic1 files
    speaker_dir = base_dir.joinpath(speaker)
    speaker_paths = list(speaker_dir.glob("**/*_mic1.flac"))

    if len(speaker_paths) < args.min:
        print("Skipping speaker {0} due to too few recordings.".format(speaker))

    if len(speaker_paths) > args.max:
        # shuffle
        random.shuffle(speaker_paths)
        speaker_paths = speaker_paths[0:args.max]

    for source_audio_path in speaker_paths:
        dest_path = out_dir.joinpath(speaker)

        # Check whether the source transcript file exists in case the transcript flag is set
        source_transcript_path = str(source_audio_path).replace("_mic1.flac",".txt")
        check_transcript_source = Path(source_transcript_path)
        if args.fetch_transcripts and not check_transcript_source.is_file():
            print("Skipping file {0} due to missing transcript.".format(source_audio_path))
            continue
        
        audio_file_name = os.path.basename(source_audio_path)
        transcript_file_name = audio_file_name.replace(".flac",".txt")

        dest_audio_file = dest_path.joinpath(audio_file_name)
        dest_transcript_file = dest_path.joinpath(transcript_file_name)

        # ensure the dir exists
        os.makedirs(dest_path, exist_ok=True)

        # Only copy files which are not needed
        check_audio = Path(dest_audio_file)
        check_transcript = Path(dest_transcript_file)

        if not check_audio.is_file():
            copyfile(source_audio_path, dest_audio_file)
        
        if args.fetch_transcripts and not check_transcript.is_file():
            copyfile(source_transcript_path, dest_transcript_file)
        
with ThreadPool(args.threads) as pool:
    list(
        tqdm(
            pool.imap(
                process_speaker,
                sorted_speakers
            ),
            "VCTK",
            len(sorted_speakers),
            unit="speakers"
        )
    )

print("Done, thanks for playing...")
