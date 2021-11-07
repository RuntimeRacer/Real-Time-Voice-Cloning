import os
from pathlib import Path
from tqdm import tqdm
import argparse
import csv
import codecs
import subprocess
import random
from multiprocess.pool import ThreadPool
from shutil import rmtree

# english
# min=4 =  20,443 speakers
# min=5 =  18,949 speakers
# min=10 = 10,884 speakers
# min=12 =  9,621 speakers
# min=14 =  8,460 speakers
# min=20 =  6,093 speakers

parser = argparse.ArgumentParser(description='Process common voice dataset for a language.')
parser.add_argument('--lang', help='Language to process', type=str)
parser.add_argument('--min', help='Minimum number of files per speaker', type=int, default=5)
parser.add_argument('--max', help='Maximum number of files per speaker', type=int, default=40)
args = parser.parse_args()

base_dir = Path("/datasets_slr/CommonVoice/{0}".format(args.lang))
clips_dir = base_dir.joinpath("clips")

print("Reading Validated.tsv file...")
speaker_hash = {}
with codecs.open(base_dir.joinpath("validated.tsv"), "r", "utf-8") as val_in:
    tsvin = csv.DictReader(val_in, delimiter='\t')
    # client_id	path	sentence	up_votes	down_votes	age	gender	accent
    # 05e9d52b02fc87f02758c2e8e1b97d05c23ec0ac7c5b76d964cb1a547ce72f7eefc021cfe23a67b34032eb931e77af13b07cde8d398660abffc411f165d24cb4	common_voice_it_17544185.mp3	Il vuoto assoluto?	2	1
    for row in tsvin:
        client_id = row["client_id"]
        if client_id not in speaker_hash:
            speaker_hash[client_id] = []

        speaker_hash[client_id].append(row["path"])
print("  - Found {} speakers...".format(len(speaker_hash)))


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

# if we have a speakers directory, remove it!
if base_dir.joinpath("speakers").is_dir() == True:
    rmtree(base_dir.joinpath("speakers"))

def process_speaker(speaker):
    # print("Processing: i: {0} - {1}".format(si, speaker))
    speaker_paths = speaker_hash[speaker]
    if len(speaker_paths) > args.max:
        # shuffle
        random.shuffle(speaker_paths)

        speaker_paths = speaker_paths[0:args.max]

    for speaker_path in speaker_paths:
        source_path = clips_dir.joinpath(speaker_path)
        # dest_path = base_dir.joinpath("speakers", str(si))
        dest_path = base_dir.joinpath("speakers", speaker[:20])

        new_name = speaker_path.replace(".mp3", "") + ".wav"
        dest_file = dest_path.joinpath(new_name)
        # print("  - Source: {0} - Dest: {1}".format(str(source_path), str(dest_file)))

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
            # "24000",
            "16000",
            str(dest_file)
        ]
        s = subprocess.call(convert_args)


with ThreadPool(8) as pool:
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
