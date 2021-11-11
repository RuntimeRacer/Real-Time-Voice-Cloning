import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import argparse
import csv
import codecs
import subprocess
import random
from multiprocess.pool import ThreadPool
from shutil import rmtree

# Functions
# _parse_speaker_data fetches validated speaker information from a .tsv file
def _parse_speaker_data(dir: Path, clips_dir: Path, lang: str) -> (Dict):
    print("Reading Validated.tsv file for language {0}...".format(lang))
    speaker_hash = {}
    with codecs.open(dir.joinpath("validated.tsv"), "r", "utf-8") as val_in:
        tsvin = csv.DictReader(val_in, delimiter='\t')
        # client_id	path	sentence	up_votes	down_votes	age	gender	accent
        # 05e9d52b02fc87f02758c2e8e1b97d05c23ec0ac7c5b76d964cb1a547ce72f7eefc021cfe23a67b34032eb931e77af13b07cde8d398660abffc411f165d24cb4	common_voice_it_17544185.mp3	Il vuoto assoluto?	2	1
        for row in tsvin:
            client_id = row["client_id"]
            if client_id not in speaker_hash:
                speaker_hash[client_id] = []

            speaker_hash[client_id].append(clips_dir.joinpath(row["path"]))
    print("  - Found {0} total speakers for language {1}.".format(len(speaker_hash), lang))
    return speaker_hash

# english CV 2
# min=4 =  20,443 speakers
# min=5 =  18,949 speakers
# min=10 = 10,884 speakers
# min=12 =  9,621 speakers
# min=14 =  8,460 speakers
# min=20 =  6,093 speakers

# Parser for Arguments
parser = argparse.ArgumentParser(description='Process common voice dataset for a language.')
parser.add_argument("datasets_root", type=Path, help=\
    "Path to the directory containing your CommonVoice datasets.")
parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument("-ar", "--audio_rate", type=int, default=16000, help=\
    "Bitrate of the output audio files. Default is 16 kHz.")
parser.add_argument('--lang', type=str, default=None, help=\
    'Language to process')
parser.add_argument('--min', type=int, default=5, help=\
    'Minimum number of files per speaker')
parser.add_argument('--max', type=int, default=40, help=\
    'Maximum number of files per speaker')
parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# Stats
speaker_count = 0
language_count = 0

# Speaker files dict
speaker_hash = {}

# Processing for a single language
if args.lang != None:
    # dirs
    base_dir = Path("{0}/{1}".format(args.datasets_root, args.lang))
    clips_dir = base_dir.joinpath("clips")

    # speaker data
    speaker_hash = _parse_speaker_data(base_dir, clips_dir, args.lang)

    # stats
    language_count += 1

# Processing for all languages
else:
    base_dir = args.datasets_root
    _, subdirs, _ = next(os.walk(base_dir))
    for lang in subdirs:
        # dirs
        sub_dir = base_dir.joinpath(lang)
        clips_dir = sub_dir.joinpath("clips")

        # speaker data
        speakers = _parse_speaker_data(sub_dir, clips_dir, lang)
        speaker_hash = {**speaker_hash, **speakers}

        # stats
        language_count += 1

speaker_count = len(speaker_hash)
print("Found {0} speakers across {1} languages.".format(speaker_count, language_count))

# Filter speakers with too small dataset
print("Pruning speakers with less than {0} files...".format(args.min))
speakers_to_remove = []
for speaker_id in speaker_hash:
    if len(speaker_hash[speaker_id]) < args.min:
        speakers_to_remove.append(speaker_id)

print("  - Pruning {0} speakers...".format(len(speakers_to_remove)))
for id in speakers_to_remove:
    del speaker_hash[id]

print("Reduced speaker pool to {0}".format(len(speaker_hash)))

# sort the speaker_id/client_id by
sorted_speakers = sorted(speaker_hash.keys())

out_dir = base_dir
if out_dir != None:
    out_dir = args.out_dir

# if we have a speakers directory, remove it!
#if out_dir.joinpath("speakers").is_dir() == True:
#    rmtree(out_dir.joinpath("speakers"))

def process_speaker(speaker):
    # print("Processing: i: {0} - {1}".format(si, speaker))
    speaker_paths = speaker_hash[speaker]
    if len(speaker_paths) > args.max:
        # shuffle
        random.shuffle(speaker_paths)

        speaker_paths = speaker_paths[0:args.max]

    for source_path in speaker_paths:
        dest_path = out_dir.joinpath("speakers", speaker)
        new_name = os.path.basename(source_path).replace(".mp3", "") + ".flac"
        dest_file = dest_path.joinpath(new_name)
        # print("  - Source: {0} - Dest: {1}".format(str(source_path), str(dest_file)))

        # ensure the dir exists
        os.makedirs(dest_path, exist_ok=True)

        # if the file already exists, skip
        check = Path(dest_file)
        if check.is_file():
            continue

        convert_args = [
            "/usr/bin/ffmpeg",
            "-y",
            "-loglevel",
            "fatal",
            "-i",
            str(source_path),
            "-ar",
            str(args.audio_rate),
            str(dest_file)
        ]
        s = subprocess.call(convert_args)


with ThreadPool(args.threads) as pool:
    list(
        tqdm(
            pool.imap(
                process_speaker,
                sorted_speakers
            ),
            args.lang,
            len(sorted_speakers),
            unit="speakers"
        )
    )

print("Done, thanks for playing...")
