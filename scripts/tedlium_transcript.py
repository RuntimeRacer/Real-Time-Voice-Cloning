import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import shutil

base_path = Path("/datasets/slr60/train-clean-360")
# speaker_dirs = [f for f in encoder_path.glob("*") if f.is_file()]
audio_files = [f for f in base_path.glob("**/*.wav")]

# loop through and group the files!
for audio_file in tqdm(audio_files):
    # new_file = base_path.joinpath(os.path.basename(speaker_dir))
    src_file = audio_file.parent.joinpath("{0}.original.txt".format(audio_file.stem))
    dest_file = audio_file.parent.joinpath("{0}.txt".format(audio_file.stem))

    # debug!
    # print("From: {0} - To: {1}".format(src_file, dest_file))
    # break

    # scary!
    # shutil.move(speaker_dir, new_dir)
    shutil.copy(src_file, dest_file)
