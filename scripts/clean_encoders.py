from pathlib import Path
from shutil import rmtree

AUTO_REMOVE_DIR = False
root_dir = Path("/output/encoder/")

speaker_dirs = [f for f in root_dir.glob("*") if f.is_dir()]
# print("Speakers: {0}".format(speaker_dirs[0:10]))

for speaker_dir in speaker_dirs:
    mels = [f for f in speaker_dir.glob("*.npy") if f.is_file()]
    # print(" - mels: {0}".format(len(mels)))
    remove_dir = False
    if len(mels) < 1:
        print(" - NO MELS! {}".format(speaker_dir))
        remove_dir = True
    elif len(mels) < 10:
        print(" - Not enough MELS ({0}) {1}".format(len(mels), speaker_dir))
        remove_dir = True

    if remove_dir == True and AUTO_REMOVE_DIR == True:
        print(" - Removing: {}".format(speaker_dir))
        # rmtree(speaker_dir)
