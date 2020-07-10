
# first merge the text files (super easy in the shell)
# export SYNTHESIZER_DEST=/output/synthesizer/train.txt
# cat /output/synthesizer_dev-clean/train.txt >> $SYNTHESIZER_DEST
# cat /output/synthesizer_train-clean-360/train.txt >> $SYNTHESIZER_DEST

from pathlib import Path
from tqdm import tqdm
import os
import shutil

merge_folders = [
    'audio',
    'embeds',
    'mels'
]
dest_path = Path("/output/synthesizer/")
source_paths = [
    Path('/output/synthesizer_dev-clean/'),
    Path('/output/synthesizer_train-clean-360/'),
]

# loop through and group the files!
for source_path in tqdm(source_paths):

    for folder in merge_folders:
        source_files = [f for f in source_path.joinpath(folder).glob("*.npy") if f.is_file()]
        for source_file in tqdm(source_files):
            dest_file = dest_path.joinpath(folder, source_file.name)
            # print("Src: {0} - Dest: {1}".format(source_file, dest_file))

            # scary!
            shutil.move(source_file, dest_file)
            # break
        # break
    # break
