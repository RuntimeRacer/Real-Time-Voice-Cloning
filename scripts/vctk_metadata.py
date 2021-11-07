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
import pandas as pd

dataset_path = Path("/datasets/VCTK-Corpus")


# read label-info
df = pd.read_table(
    dataset_path.joinpath("speaker-info.txt"),
    # usecols=['ID'],
    index_col=False,
    delim_whitespace=True
)

speaker_hash = {}
# ID  AGE  GENDER  ACCENTS  REGION
# 225  23  F    English    Southern  England
# 226  22  M    English    Surrey
# 227  38  M    English    Cumbria
for row in df.values:
    speaker_id = str(row[0])

    age = row[1]
    if age > 99:
        age = "centenarian"
    elif age >= 90 and age < 100:
        age = "nineties"
    elif age >= 80 and age < 90:
        age = "eighties"
    elif age >= 70 and age < 80:
        age = "seventies"
    elif age >= 60 and age < 70:
        age = "sixties"
    elif age >= 50 and age < 60:
        age = "fifties"
    elif age >= 40 and age < 50:
        age = "fourties"
    elif age >= 30 and age < 40:
        age = "thirties"
    elif age >= 20 and age < 30:
        age = "twenties"
    elif age >= 13 and age < 20:
        age = "teens"
    elif age < 13:
        age = "child"
    else:
        age = "unknown"

    speaker_hash[speaker_id] = {
        "id": speaker_id,
        "gender": "male" if row[2].lower() == "m" else "female",
        "age": age,
        "age_year": row[1],
        "accent": row[3],
        "region": row[4],
    }

base_dir = Path("/output/encoder/")

# find our speaker dirs
speaker_dirs = [f for f in base_dir.glob("VCTK-Corpus*") if f.is_dir()]
speaker_dirs.sort()

for speaker_dir in speaker_dirs:
    metadata = {
        "gender": "unknown",
        "age": "unknown",
        "age_year": "unknown",
        "accent": "unknown",
        "region": "unknown",
        "language": "en",
        "utterances": {},
    }

    speaker_id = speaker_dir.name.split("_")[-1][1:]

    if speaker_id in speaker_hash:
        speaker_data = speaker_hash[speaker_id]
        metadata["gender"] = speaker_data["gender"]
        metadata["age"] = speaker_data["age"]
        metadata["age_year"] = speaker_data["age_year"]
        metadata["accent"] = speaker_data["accent"]
        metadata["region"] = speaker_data["region"]

    # print("metadata: {}".format(metadata))

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
