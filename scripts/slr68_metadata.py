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

dataset_path = Path("/datasets/slr68")

speaker_hash = {}
with codecs.open(dataset_path.joinpath("SPKINFO.txt"), "r", "utf-8") as val_in:
    tsvin = csv.reader(val_in, delimiter='\t')

    # SPKID	Age	Gender	Dialect
    # 14_3466	18	female	he bei
    for i, row in enumerate(tsvin):
        if i == 0:
            continue

        speaker_hash[row[0]] = {
            "id": row[0],
            "gender": row[2].lower(),
            "nationality": row[3],
            "age": int(row[1])
        }

base_dir = Path("/output/encoder/")

# find our speaker dirs
speaker_dirs = [f for f in base_dir.glob("slr68_*") if f.is_dir()]
speaker_dirs.sort()

for speaker_dir in speaker_dirs:
    metadata = {
        "gender": "unknown",
        "age": "unknown",
        "accent": "unknown",
        "language": "zh",
        "utterances": {},
    }

    speaker_id = "_".join(speaker_dir.name.split("_")[-2:])
    # print("speaker: {} - {}".format(speaker_dir.name, speaker_id))
    # sys.exit(1)

    if speaker_id in speaker_hash:
        speaker_data = speaker_hash[speaker_id]
        metadata["gender"] = speaker_data["gender"]
        metadata["nationality"] = speaker_data["nationality"]

        # parse the age
        # ['eighties' 'fifties' 'fourties' 'nineties' 'seventies' 'sixties' 'teens'
        #  'thirties' 'twenties' 'unknown']
        age = speaker_data["age"]
        out_age = "unknown"
        if age >= 90 and age < 100:
            out_age = "nineties"
        elif age >= 80 and age < 90:
            out_age = "eighties"
        elif age >= 70 and age < 80:
            out_age = "seventies"
        elif age >= 60 and age < 70:
            out_age = "sixties"
        elif age >= 50 and age < 60:
            out_age = "fifties"
        elif age >= 40 and age < 50:
            out_age = "fourties"
        elif age >= 30 and age < 40:
            out_age = "thirties"
        elif age >= 20 and age < 30:
            out_age = "twenties"
        elif age >= 13 and age < 20:
            out_age = "teens"

        # update the metadata
        metadata["age"] = out_age

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
