# use a different docker image!
# make build_align && make run_align

import sys
import json
import csv
from pathlib import Path

DATA_TYPE = "phones"

source_path = Path("/output/montreal-aligned/stats_dev.json")
dest_path = source_path \
    .with_name("{}_{}".format(source_path.stem, DATA_TYPE)) \
    .with_suffix(".csv")

with open(source_path, "r", encoding="utf8") as file_in:
    data = json.load(file_in)


with open(dest_path, "w", encoding="utf8") as file_out:
    # we only care about phones!
    data = data[DATA_TYPE]

    # create a csv writer (dict)
    csv_writer = csv.DictWriter(
        file_out,
        [
            "text",
            "count",
            "duration_min",
            "duration_max",
            "duration_avg",
            "duration_std",
        ]
    )

    # write our header
    csv_writer.writeheader()

    for item_key in data:
        # get the actual item
        item = data[item_key]

        # write it
        csv_writer.writerow({
            "text": item["text"],
            "count": item["count"],
            "duration_min": item["duration"]["min"],
            "duration_max": item["duration"]["max"],
            "duration_avg": item["duration"]["avg"],
            "duration_std": item["duration"]["std"],
        })

print("All done, thanks for playing!")
