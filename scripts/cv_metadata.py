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
import sys
import numpy as np
import json

parser = argparse.ArgumentParser(description='Process common voice dataset for a language.')
parser.add_argument('--lang', help='Language to process', type=str)
args = parser.parse_args()

base_dir = Path("/datasets/CommonVoice/{0}".format(args.lang))
dest_dir = "/output/encoder/CommonVoice_{0}_speakers".format(args.lang)

print("Reading Validated.tsv file {}...".format(args.lang))
speaker_hash = {}
with codecs.open(base_dir.joinpath("validated.tsv"), "r", "utf-8") as val_in:
    tsvin = csv.DictReader(val_in, delimiter='\t')

    # client_id	path	sentence	up_votes	down_votes	age	gender	accent
    # 05e9d52b02fc87f02758c2e8e1b97d05c23ec0ac7c5b76d964cb1a547ce72f7eefc021cfe23a67b34032eb931e77af13b07cde8d398660abffc411f165d24cb4	common_voice_it_17544185.mp3	Il vuoto assoluto?	2	1
    for row in tsvin:
        client_id = row["client_id"][0:20]
        if client_id not in speaker_hash:
            speaker_hash[client_id] = {
                "client_id": [],
                "path": [],
                "sentence": [],
                "up_votes": [],
                "down_votes": [],
                "age": [],
                "gender": [],
                "accent": [],
            }

        for k in row.keys():
            if k in speaker_hash[client_id]:
                speaker_hash[client_id][k].append(row[k])

# print("speaker_hash: {}".format(speaker_hash))
print("  - Found {} speakers...".format(len(speaker_hash)))

# loop through each speaker and find the dest dir
for client_id in speaker_hash.keys():
    speaker = speaker_hash[client_id]
    # print("Processing: {}".format(client_id))

    # ensure this speaker exists
    speaker_dir = Path("{}_{}".format(dest_dir, client_id))
    if speaker_dir.exists() == False:
        continue

    # build our base metadata
    metadata = {
        "age": None,
        "gender": None,
        "language": args.lang,
        "accent": None,
        "utterances": {}
    }

    # update the easy ones
    for itm in ["age", "gender", "accent"]:
        values, counts = np.unique(speaker[itm], return_counts=True)
        metadata[itm] = values[np.argmax(counts)]
        if metadata[itm] == "" or metadata[itm] is None:
            metadata[itm] = "unknown"

    # get the utterance data
    for path, sentence, up_votes, down_votes in zip(speaker["path"], speaker["sentence"], speaker["up_votes"], speaker["down_votes"]):
        # remove ".mp3" from path
        path = path.replace(".mp3", "")
        if Path(speaker_dir.joinpath("{}.npy".format(path))).exists() == False:
            continue

        metadata["utterances"][path] = {
            "path": path,
            "sentence": sentence,
            "up_votes": int(up_votes),
            "down_votes": int(down_votes),
        }

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
