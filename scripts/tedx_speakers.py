import os
from pathlib import Path
import webvtt
import argparse
import sox
import random
from tqdm import tqdm
from multiprocess.pool import ThreadPool

# TODO: Make this capable of multilang processing like CV-pre-pre-processor
# Parser for Arguments
parser = argparse.ArgumentParser(description='Process Multilingual TEDx.')
parser.add_argument("datasets_root", type=Path, help=\
    "Path to the directory containing your CommonVoice datasets.")
parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument('--min', type=int, default=5, help=\
    'Minimum number of files per speaker')
parser.add_argument('--max', type=int, default=40, help=\
    'Maximum number of files per speaker')
parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# dirs
base_dir = args.datasets_root
wav_dir = base_dir.joinpath("wav") # Contains WAV / FLAC files (Audio)
vtt_dir = base_dir.joinpath("vtt") # Contains VTT files (Text alignments)
out_dir = base_dir.joinpath("speakers")
if out_dir != None:
    out_dir = args.out_dir

# Process files
source_files = [f for f in wav_dir.glob("*.flac") if f.is_file()]
sorted_files = sorted(source_files)

# Process individual files in threadpool
def process_file(file):
    # file details
    name = file.name
    file_name = file.stem
    suffix = file.suffix
    # print("Processing: {0}...".format(name))

    # Get matching VTT file for this audio file and retrieve the segments    
    vtt_path = next(vtt_dir.glob("{}*.vtt".format(file_name)))
    vtt_segments = list(webvtt.read(vtt_path))

    if len(vtt_segments) < args.min:
        print("Skipping speaker {0} due to too few recordings.".format(file_name))

    if len(vtt_segments) > args.max:
        # shuffle
        random.shuffle(vtt_segments)
        vtt_segments = vtt_segments[0:args.max]

    # Make sure speaer dir exists
    out_path = out_dir.joinpath(file_name)
    os.makedirs(out_path, exist_ok=True)

    # process all segments for this speaker
    for si, segment in enumerate(vtt_segments):
        # define output file
        out_file = Path(out_path).joinpath("{0}_{1:04d}.wav".format(file_name, si))

        # Initialize Transformer
        transformer = sox.Transformer()
        transformer.trim(segment.start_in_seconds, segment.end_in_seconds)
        transformer.build(str(file), str(out_file))


with ThreadPool(args.threads) as pool:
    list(
        tqdm(
            pool.imap(
                process_file,
                sorted_files
            ),
            "Multilingual TEDx",
            len(sorted_files),
            unit="speakers"
        )
    )

print("Done, thanks for playing...")