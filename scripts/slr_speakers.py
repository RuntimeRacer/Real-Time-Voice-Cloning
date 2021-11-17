import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import argparse
import random
from multiprocess.pool import ThreadPool
from shutil import copyfile

# Functions
def mapFilesToSpeakers(dir):
    # Dict for speakers
    speakers = {}

    # Get all files in directory
    files = list(dir.glob("**/*.wav"))

    # Iterate through files and figure out speakes based on naming patter
    for file in files:
        file_name = os.path.basename(file)
        name_parts = file_name.split("_")
        speaker_id = "{0}_{1}".format(name_parts[0], name_parts[1])
        
        if not speaker_id in speakers:
            speakers[speaker_id] = []
        
        speakers[speaker_id].append(file)
        
    # Return speaker map
    return speakers


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
parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# Stats
speaker_count = 0

# Get speakers
base_dir = args.datasets_root
speakers = mapFilesToSpeakers(base_dir)

speaker_count = len(speakers)
print("Found {0} speakers.".format(speaker_count))

# sort the speaker_id/client_id by
sorted_speakers = sorted(speakers.keys())

out_dir = base_dir
if out_dir != None:
    out_dir = args.out_dir

# if we have a speakers directory, remove it!
#if out_dir.joinpath("speakers").is_dir() == True:
#    rmtree(out_dir.joinpath("speakers"))

def process_speaker(speaker_id):
    # Get list of files for each speaker
    speaker_paths = speakers[speaker_id]

    if len(speaker_paths) < args.min:
        print("Skipping speaker {0} due to too few recordings.".format(speaker_id))

    if len(speaker_paths) > args.max:
        # shuffle
        random.shuffle(speaker_paths)
        speaker_paths = speaker_paths[0:args.max]

    for source_path in speaker_paths:
        dest_path = out_dir.joinpath(speaker_id)
        file_name = os.path.basename(source_path)
        dest_file = dest_path.joinpath(file_name)
        # print("  - Source: {0} - Dest: {1}".format(str(source_path), str(dest_file)))

        # ensure the dir exists
        os.makedirs(dest_path, exist_ok=True)

        # if the file already exists, skip
        check = Path(dest_file)
        if check.is_file():
            continue

        # Copy the files
        copyfile(source_path, dest_file)

with ThreadPool(args.threads) as pool:
    list(
        tqdm(
            pool.imap(
                process_speaker,
                sorted_speakers
            ),
            "SLR-Speakers",
            len(sorted_speakers),
            unit="speakers"
        )
    )

print("Done, thanks for playing...")
