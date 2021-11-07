import os
from pathlib import Path
from tqdm import tqdm
import argparse
import csv
import codecs
import subprocess
import random
import sys

parser = argparse.ArgumentParser(description='Process common voice dataset for a language.')
parser.add_argument('--lang', help='Language to process', type=str)
# parser.add_argument('--min', help='Minimum number of files per speaker', type=int, default=5)
# parser.add_argument('--max', help='Maximum number of files per speaker', type=int, default=40)
args = parser.parse_args()

base_dir = Path("/datasets/CommonVoice/{0}".format(args.lang))
clips_dir = base_dir.joinpath("clips")

print("Reading Validated.tsv file...")
speaker_hash = {}
with codecs.open(base_dir.joinpath("validated.tsv"), "r", "utf-8") as val_in:
    tsvin = csv.DictReader(val_in, delimiter='\t')
    # client_id	path	sentence	up_votes	down_votes	age	gender	accent
    # 05e9d52b02fc87f02758c2e8e1b97d05c23ec0ac7c5b76d964cb1a547ce72f7eefc021cfe23a67b34032eb931e77af13b07cde8d398660abffc411f165d24cb4	common_voice_it_17544185.mp3	Il vuoto assoluto?	2	1
    for row in tsvin:
        client_id = row["client_id"][0:20]
        if client_id not in speaker_hash:
            speaker_hash[client_id] = []

        speaker_hash[client_id].append(row)
print("  - Found {} speakers...".format(len(speaker_hash)))

# sort the speaker_id/client_id by
sorted_speakers = sorted(speaker_hash.keys())

def process_speaker(speaker_client_id):
    speaker = speaker_hash[speaker_client_id]
    # print("Processing: i: {0} - {1}".format(speaker_client_id, speaker))

    for utterance in speaker:
        utterance_path = base_dir.joinpath(
            "speakers",
            speaker_client_id,
            utterance["path"].replace(".mp3", ".wav")
        )

        # create a txt file with the sentence if the wav file exists
        if utterance_path.exists():
            sentence_path = utterance_path.with_suffix(".txt")
            if not sentence_path.exists():
                with open(sentence_path, "w", encoding="utf8") as out:
                    out.write(utterance["sentence"])

for id in tqdm(sorted_speakers, unit="speakers"):
    process_speaker(id)

print("Done, thanks for playing...")
