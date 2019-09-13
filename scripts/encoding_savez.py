import numpy as np
from pathlib import Path
from tqdm import tqdm

encoder_path = Path("/output/encoder")
speaker_dirs = [f for f in encoder_path.glob("*") if f.is_dir()]

# loop through and group the files!
for speaker_dir in tqdm(speaker_dirs):
    files = [f for f in speaker_dir.glob("*.npy") if f.is_file()]

    np_args = {}
    for file in files:
        np_file = np.load(file)
        np_args[file.name] = np_file

    speaker_combined = speaker_dir.joinpath("combined.npz")

    np.savez(speaker_combined, **np_args)


# read the file example...
# with np.load('/output/encoder/VoxCeleb2_dev_aac_id02942/combined.npz') as data:
#     print(data["pPn0ccfGgeg_00014.npy"])
