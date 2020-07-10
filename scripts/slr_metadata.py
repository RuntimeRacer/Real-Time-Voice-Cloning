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

# which datasets make the gender part easy?
slr_valid = [
    "41", "42", "43", "44", "61", "63", "64", "65", "66", "69", "70",
    "71", "72", "73", "74", "75", "76", "77", "78", "79", "80"
]

language_mapping = {
    "41": "jv",
    "42": "km",
    "43": "ne",
    "44": "su",
    "61": "es",
    "63": "ml",
    "64": "mr",
    "65": "ta",
    "66": "te",
    "69": "ca",
    "70": "en-NG",
    "71": "es-CL",
    "72": "es-CO",
    "73": "es-PE",
    "74": "es-PR",
    "75": "es-VE",
    "76": "eu-ES",
    "77": "gl-ES",
    "78": "gu",
    "79": "kn",
    "80": "my",
}

base_dir = Path("/output/encoder/")

# find our speaker dirs
speaker_dirs = [f for f in base_dir.glob("slr*") if f.is_dir()]
speaker_dirs.sort()
print("dirs: {}".format(len(speaker_dirs)))

# filter the dirs
speaker_dirs_filtered = [f for f in speaker_dirs if f.name[3:5] in slr_valid]

for speaker_dir in speaker_dirs_filtered:
    metadata = {
        "gender": "unknown",
        "age": "unknown",
        "accent": "unknown",
        "language": language_mapping[speaker_dir.name[3:5]],
        "utterances": {},
    }

    if "_female_" in speaker_dir.name:
        metadata["gender"] = "female"
    elif "_male_" in speaker_dir.name:
        metadata["gender"] = "male"

    # print("speaker: {}".format(speaker_dir.name))

    # save our output
    with open(speaker_dir.joinpath("metadata.json"), 'w', encoding='utf8') as outfile:
        json.dump(metadata, outfile, indent=4)


print("Done, thanks for playing...")
