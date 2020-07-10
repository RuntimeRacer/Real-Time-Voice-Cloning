import os
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
import random

# should pull this from args
parser = argparse.ArgumentParser(description='Process TIMIT dataset.')
parser.add_argument('--train', help='TRAIN if true else TEST', action='store_true')
args = parser.parse_args()

# little debugging can't hurt
print("Processing {} files...".format("TRAIN" if args.train else "TEST"))

# build our base dir
base_dir = Path("/datasets/TIMIT/data/{}".format("TRAIN" if args.train else "TEST"))
output_dir = Path("/datasets/TIMIT/speakers")

# find our audio files
print("Searching for all wav files...")
source_files = [f for f in base_dir.glob("**/*.wav") if f.is_file()]
print("  - Found: {}".format(len(source_files)))

# group files based on speaker id r0000000
speaker_hash = {}
for file in source_files:
    client_id = file.parent.stem

    if client_id not in speaker_hash:
        speaker_hash[client_id] = []

    speaker_hash[client_id].append(file)

print("Found {} unique speakers".format(len(speaker_hash)))

# sort the speaker_id/client_id by
sorted_speakers = sorted(speaker_hash.keys())

for speaker in tqdm(sorted_speakers):
    # print("Processing: i: {0} - {1}".format(si, speaker))
    speaker_paths = speaker_hash[speaker]
    for speaker_path in speaker_paths:
        source_path = speaker_path
        dest_path = output_dir.joinpath(speaker)

        new_file_name = speaker_path.name.replace('.WAV', '')
        dest_file = dest_path.joinpath(new_file_name)
        # print("  - Source: {0} - Dest: {1}".format(str(source_path), str(dest_file)))
        # break

        # ensure the dir exists
        os.makedirs(dest_path, exist_ok=True)

        shutil.copyfile(source_path, dest_file)
        # break
    # break

print("Done, thanks for playing...")
