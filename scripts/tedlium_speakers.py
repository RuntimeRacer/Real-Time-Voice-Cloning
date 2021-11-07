import os
from pathlib import Path
from stm import parse_stm_file
import sox
from tqdm import tqdm

out_dir = Path("/datasets_slr/TEDLIUM_release-3/data/speakers")
wav_dir = Path("/datasets_slr/TEDLIUM_release-3/data/wav")
stm_dir = Path("/datasets_slr/TEDLIUM_release-3/data/stm")

source_files = [f for f in wav_dir.glob("*.wav") if f.is_file()]

for file in tqdm(source_files):
    name = file.name
    stem = file.stem
    suffix = file.suffix
    # print("Processing: {0}...".format(name))

    # ensure the output path exists for this speaker!
    out_path = out_dir.joinpath(stem.split('_')[0])
    try:
        os.makedirs(out_path)
    except FileExistsError:
        # directory already exists
        pass

    stm_path = stm_dir.joinpath("{}.stm".format(stem))

    stm_segments = parse_stm_file(stm_path)
    for si, segment in enumerate(stm_segments):
        out_file = Path(out_path).joinpath("{0}_{1:04d}.wav".format(stem, si))
        # print(segment.transcript)

        tfm = sox.Transformer()
        tfm.trim(segment.start_time, segment.stop_time)
        tfm.build(str(file), str(out_file))
        # break

    # break
