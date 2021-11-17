import os
from pathlib import Path
import argparse
from tqdm import tqdm
import random
from multiprocess.pool import ThreadPool
from shutil import copyfile

# should pull this from args
parser = argparse.ArgumentParser(description='Process nasjonalbank dataset for a language.')
parser.add_argument("datasets_root", type=Path, help=\
    "Path to the directory containing your CommonVoice datasets.")
parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument('--lang', help=\
    "Language to process", type=str)
parser.add_argument('--min', help=\
    "Minimum number of files per speaker", type=int, default=12)
parser.add_argument('--max', help=\
    "Maximum number of files per speaker", type=int, default=40)
parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# Stats
speaker_count = 0
language_count = 0

# Processing for a single language
if args.lang != None:
    # dirs
    base_dir = Path("{0}/{1}".format(args.datasets_root, args.lang))
else:
    base_dir = args.datasets_root

# build our output dir
out_dir = base_dir
if out_dir != None:
    out_dir = args.out_dir

# find our audio files
print("Searching for all wav files...")
source_files = [f for f in base_dir.glob("**/*.wav") if f.is_file()]
print("  - Found: {}".format(len(source_files)))

# group files based on speaker id r0000000
speaker_hash = {}
for file in source_files:
    client_id = "{0}_{1}".format(file.parts[-3], file.parts[-2])

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

def process_speaker(speaker):
    # print("Processing: i: {0} - {1}".format(si, speaker))
    speaker_paths = speaker_hash[speaker]
    if len(speaker_paths) > args.max:
        # shuffle
        random.shuffle(speaker_paths)

        speaker_paths = speaker_paths[0:args.max]

    for source_path in speaker_paths:
        dest_path = out_dir.joinpath("speakers", speaker)
        new_name = os.path.basename(source_path)
        dest_file = dest_path.joinpath(new_name)
        # print("  - Source: {0} - Dest: {1}".format(str(source_path), str(dest_file)))
        # break

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
            "Nasjonalbank",
            len(sorted_speakers),
            unit="speakers"
        )
    )

print("Done, thanks for playing...")
