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

DATASETS = [
    "cv-en",
    # 'dev-clean',
    # 'dev-other',
    # 'test-clean',
    # 'test-other',
    # 'train-clean-100',
    # 'train-clean-360',
    # 'train-other-500',
]

# shared across ALL of our datasets
dataset_phones = {}
dataset_words = {}

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

# process each dataset!
for DATASET in DATASETS:
    base_path = Path('/output/montreal-aligned/{}'.format(DATASET))
    speaker_dirs = [f for f in base_path.glob("*") if f.is_dir()]

    for speaker_dir in tqdm(speaker_dirs, desc=DATASET):
        textgrid_files = [f for f in speaker_dir.glob("*.TextGrid") if f.is_file()]

        if len(textgrid_files) == 0:
            book_dirs = [f for f in speaker_dir.glob("*") if f.is_dir()]
            for book_dir in book_dirs:
                # find our textgrid files
                for f in book_dir.glob("*.TextGrid"):
                    if not f.is_file():
                        continue
                    textgrid_files.append(f)

        # sort them!
        textgrid_files = sorted(textgrid_files)

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

# update our duration data to store just the stats
duration_stats(dataset_words)
duration_stats(dataset_phones)

# save the data!
with open(Path('/output/montreal-aligned/').joinpath("stats.json"), "w") as json_out:
    json.dump(
        {
            "words": dataset_words,
            "phones": dataset_phones,
        },
        json_out,
        indent=4
    )

print("All done, thanks for playing!")
