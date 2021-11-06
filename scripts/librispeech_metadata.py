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


speaker_hash = {}
with codecs.open(Path("/datasets/slr60/speakers.tsv"), "r", "utf-8") as val_in:
    tsvin = csv.DictReader(val_in, delimiter='\t')

    # READER	GENDER	SUBSET NAME
    # 14	F	train-clean-360	Kristin LeMoine
    for row in tsvin:
        speaker_hash[row["READER"]] = {
            "id": row["READER"],
            "gender": "male" if row["GENDER"].lower() == "m" else "female",
            "subset": row["SUBSET"],
            "name": row["NAME"],
        }

base_dir = Path("/output/encoder/")

# find our speaker dirs
speaker_dirs = [f for f in base_dir.glob("slr60_*") if f.is_dir()]
speaker_dirs.sort()

for speaker_dir in speaker_dirs:
    metadata = {
        "gender": "unknown",
        "age": "unknown",
        "accent": "unknown",
        "language": "en",
        "utterances": {},
    }

    # if "_female_" in speaker_dir.name:
    #     metadata["gender"] = "female"
    # elif "_male_" in speaker_dir.name:
    #     metadata["gender"] = "male"
    speaker_id = speaker_dir.name.split("_")[-1]

    if speaker_id in speaker_hash:
        speaker_data = speaker_hash[speaker_id]
        metadata["gender"] = speaker_data["gender"]
        metadata["name"] = speaker_data["name"]

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
