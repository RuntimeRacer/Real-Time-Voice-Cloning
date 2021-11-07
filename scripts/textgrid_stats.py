# use a different docker image!
# make build_align && make run_align
# bin/mfa_align \
#   /datasets/CommonVoice/en/speakers \
#   /datasets/slr60/english.dict \
#   /opt/Montreal-Forced-Aligner/dist/montreal-forced-aligner/pretrained_models/english.zip \
#   /output/montreal-aligned/cv-en/
# bin/mfa_validate_dataset \
#   /datasets/slr60/test-clean \
#   /datasets/slr60/english.dict\
#   english

import sys
import tgt
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

DATASET = 'dev-clean'
# DATASET = 'dev-other'
# DATASET = 'test-clean'
# DATASET = 'test-other'
# DATASET = 'train-clean-100'
# DATASET = 'train-clean-360'
# DATASET = 'train-other-500'
base_path = Path('/output/montreal-aligned/{}'.format(DATASET))
speaker_dirs = [f for f in base_path.glob("*") if f.is_dir()]

dataset_phones = {}
dataset_words = {}

for speaker_dir in tqdm(speaker_dirs):
    book_dirs = [f for f in speaker_dir.glob("*") if f.is_dir()]
    for book_dir in book_dirs:
        # find our textgrid files
        textgrid_files = sorted([f for f in book_dir.glob("*.TextGrid") if f.is_file()])

        # process each grid file and add to our output
        for textgrid_file in textgrid_files:
            # read the grid
            input = tgt.io.read_textgrid(textgrid_file)
            # print("input: {}".format(input))
            # sys.exit(1)

            # get all the word tiers
            word_tier = input.get_tier_by_name('words')
            phone_tier = input.get_tier_by_name('phones')

            for interval in word_tier:
                if interval.text not in dataset_words:
                    dataset_words[interval.text] = {
                        "text": interval.text,
                        "count": 0,
                        "duration": []
                    }

                # increase the count
                dataset_words[interval.text]["count"] += 1

                # add to our duration
                dataset_words[interval.text]["duration"].append(
                    interval.end_time - interval.start_time
                )

            for interval in phone_tier:
                if interval.text not in dataset_phones:
                    dataset_phones[interval.text] = {
                        "text": interval.text,
                        "count": 0,
                        "duration": []
                    }

                # increase the count
                dataset_phones[interval.text]["count"] += 1

                # add to our duration
                dataset_phones[interval.text]["duration"].append(
                    interval.end_time - interval.start_time
                )

def duration_stats(dataset):
    for k in dataset.keys():
        vals = np.array(dataset[k]["duration"])
        dataset[k]["duration"] = {
            "min": np.min(vals),
            "max": np.max(vals),
            "avg": np.mean(vals),
            "std": np.std(vals)
        }

    return dataset

duration_stats(dataset_words)
duration_stats(dataset_phones)

with open(base_path.joinpath("stats.json"), "w") as json_out:
    json.dump(
        {
            "words": dataset_words,
            "phones": dataset_phones,
        },
        json_out,
        indent=4
    )

print("All done, thanks for playing!")
