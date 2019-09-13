import os
from pathlib import Path
import argparse
from tqdm import tqdm
import subprocess
import random

# should pull this from args
parser = argparse.ArgumentParser(description='Process common voice dataset for a language.')
parser.add_argument('--lang', help='Language to process', type=str)
parser.add_argument('--min', help='Minimum number of files per speaker', type=int, default=12)
parser.add_argument('--max', help='Maximum number of files per speaker', type=int, default=40)
args = parser.parse_args()

# little debugging can't hurt
print("Processing {} language files...".format(args.lang))

# build our base dir
base_dir = Path("/datasets/nasjonal-bank/{}".format(args.lang))
speakers_dir = base_dir.joinpath("speakers")

# find our audio files
print("Searching for all wav files...")
source_files = [f for f in base_dir.glob("**/*.wav") if f.is_file()]
print("  - Found: {}".format(len(source_files)))

# group files based on speaker id r0000000
speaker_hash = {}
for file in source_files:
    client_id = file.parts[-2]

    if client_id not in speaker_hash:
        speaker_hash[client_id] = []

    speaker_hash[client_id].append(file)

print("Found {} unique speakers".format(len(speaker_hash)))

print("Pruning speakers with less than {} files...".format(args.min))
speakers_to_remove = []
for speaker_id in speaker_hash:
    if len(speaker_hash[speaker_id]) < args.min:
        speakers_to_remove.append(speaker_id)

print("  - Pruning {} speakers...".format(len(speakers_to_remove)))
for id in speakers_to_remove:
    del speaker_hash[id]

print("Reduced speaker pool to {}".format(len(speaker_hash)))

# sort the speaker_id/client_id by
sorted_speakers = sorted(speaker_hash.keys())

for speaker in tqdm(sorted_speakers):
    # print("Processing: i: {0} - {1}".format(si, speaker))
    speaker_paths = speaker_hash[speaker]
    if len(speaker_paths) > args.max:
        # shuffle
        random.shuffle(speaker_paths)

        speaker_paths = speaker_paths[0:args.max]

    for speaker_path in speaker_paths:
        source_path = speaker_path
        dest_path = speakers_dir.joinpath(speaker)

        # new_name = speaker_path.replace(".mp3", "") + ".wav"
        new_name = speaker_path.name
        dest_file = dest_path.joinpath(new_name)
        # print("  - Source: {0} - Dest: {1}".format(str(source_path), str(dest_file)))
        # break

        # ensure the dir exists
        os.makedirs(dest_path, exist_ok=True)

        convert_args = [
            "/usr/bin/ffmpeg",
            "-y",
            "-loglevel",
            "fatal",
            "-i",
            str(source_path),
            "-ar",
            "16000",
            '-ac',
            '1',
            str(dest_file)
        ]
        s = subprocess.call(convert_args)

    #     break
    # break

print("Done, thanks for playing...")
