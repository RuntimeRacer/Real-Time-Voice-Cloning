import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import shutil

base_path = Path("/output/sv2tts_mixed_768_relu/encoder")
encoder_path = Path("/output/sv2tts_english/encoder")
speaker_dirs = [f for f in encoder_path.glob("*") if f.is_dir()]

# loop through and group the files!
for speaker_dir in tqdm(speaker_dirs):
    new_dir = base_path.joinpath(os.path.basename(speaker_dir))

    # debug!
    # print("From: {0} - To: {1}".format(speaker_dir, new_dir))
    # break

    # scary!
    shutil.move(speaker_dir, new_dir)



# read the file example...
# with np.load('/output/encoder/VoxCeleb2_dev_aac_id02942/combined.npz') as data:
#     print(data["pPn0ccfGgeg_00014.npy"])
