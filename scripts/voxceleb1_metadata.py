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

dataset_path = Path("/datasets/voxceleb/VoxCeleb1")

speaker_hash = {}
with codecs.open(dataset_path.joinpath("vox1_meta.csv"), "r", "utf-8") as val_in:
    tsvin = csv.reader(val_in, delimiter='\t')

    # ['VoxCeleb1 ID', 'VGGFace1 ID', 'Gender', 'Nationality', 'Set']
    for i, row in enumerate(tsvin):
        if i == 0:
            continue

        speaker_hash[row[0]] = {
            "id": row[0],
            "gender": "male" if row[2].lower() == "m" else "female",
            "name": row[1],
            "nationality": row[3],
            "set": row[4],
        }

base_dir = Path("/output/encoder/")

# find our speaker dirs
speaker_dirs = [f for f in base_dir.glob("voxceleb_VoxCeleb1_*") if f.is_dir()]
speaker_dirs.sort()

for speaker_dir in speaker_dirs:
    metadata = {
        "gender": "unknown",
        "age": "unknown",
        "accent": "unknown",
        "language": "en",
        "utterances": {},
    }

    speaker_id = speaker_dir.name.split("_")[-1]

    if speaker_id in speaker_hash:
        speaker_data = speaker_hash[speaker_id]
        metadata["gender"] = speaker_data["gender"]
        metadata["name"] = speaker_data["name"]
        metadata["nationality"] = speaker_data["nationality"]

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
