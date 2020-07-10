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

dataset_path = Path("/datasets/voxceleb/VoxCeleb2")

speaker_hash = {}
with codecs.open(dataset_path.joinpath("vox2_meta.csv"), "r", "utf-8") as val_in:
    tsvin = csv.reader(val_in, delimiter=',')

    # VoxCeleb2 ID ,VGGFace2 ID ,Gender ,Set
    for i, row in enumerate(tsvin):
        if i == 0:
            continue

        speaker_hash[row[0].strip()] = {
            "id": row[0].strip(),
            "gender": "male" if row[2].strip().lower() == "m" else "female",
            "set": row[3],
        }

base_dir = Path("/output/encoder/")

# find our speaker dirs
speaker_dirs = [f for f in base_dir.glob("voxceleb_VoxCeleb2_*") if f.is_dir()]
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

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
