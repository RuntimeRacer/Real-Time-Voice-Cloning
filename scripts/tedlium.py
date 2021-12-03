import os
from pathlib import Path
from stm import parse_stm_file
import argparse
import sox
import random
from tqdm import tqdm
from multiprocess.pool import ThreadPool

# Parser for Arguments
parser = argparse.ArgumentParser(description='Process TED-LIUM v3.')
parser.add_argument("datasets_root", type=Path, help=\
    "Path to the directory containing your CommonVoice datasets.")
parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument('--min', type=int, default=12, help=\
    'Minimum number of files per speaker')
parser.add_argument('--max', type=int, default=40, help=\
    'Maximum number of files per speaker')
parser.add_argument("-ft", "--fetch_transcripts", type=bool, default=False, help=\
    "Path to the ouput directory for this preprocessing script")
parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# dirs
base_dir = args.datasets_root
# wav_dir = base_dir.joinpath("wav") # Contains WAV files (Audio, outdated)
sph_dir = base_dir.joinpath("sph") # Contains SPH files (Audio)
stm_dir = base_dir.joinpath("stm") # Contains STM file (Metadata)
out_dir = base_dir.joinpath("speakers")
if out_dir != None:
    out_dir = args.out_dir

# Process files
source_files = [f for f in sph_dir.glob("*.sph") if f.is_file()]
sorted_files = sorted(source_files)

# Process individual files in threadpool
def process_file(file):
    # file details
    name = file.name
    file_name = file.stem
    suffix = file.suffix
    # print("Processing: {0}...".format(name))

    # Get ID for this speaker
    speaker_id = out_dir.joinpath(file_name.split('_')[0])    

    # Get matching STM file for this audio file and retrieve the segments
    stm_path = stm_dir.joinpath("{}.stm".format(file_name))
    stm_segments = parse_stm_file(stm_path)

    if len(stm_segments) < args.min:
        print("Skipping speaker {0} due to too few recordings.".format(speaker_id))
        return

    if len(stm_segments) > args.max:
        # shuffle
        random.shuffle(stm_segments)
        stm_segments = stm_segments[0:args.max]

    # Make sure speaker dir exists
    out_path = out_dir.joinpath(speaker_id)
    os.makedirs(out_path, exist_ok=True)

    # process all segments for this speaker
    for si, segment in enumerate(stm_segments):
        # define output file
        out_file = Path(out_path).joinpath("{0}_{1:04d}.wav".format(file_name, si))

        if args.fetch_transcripts:
            out_file_transcript = Path(out_path).joinpath("{0}_{1:04d}.txt".format(file_name, si))

             # Get transcript and remove unwanted content
            transcript = segment.transcript
            transcript = transcript.replace("<unk>","")
            transcript = transcript.strip()

            if not out_file_transcript.exists():
                with open(out_file_transcript, "w", encoding="utf8") as out:
                    out.write(transcript)

        # Initialize Transformer
        transformer = sox.Transformer()
        transformer.trim(segment.start_time, segment.stop_time)
        transformer.build(str(file), str(out_file))


with ThreadPool(args.threads) as pool:
    list(
        tqdm(
            pool.imap(
                process_file,
                sorted_files
            ),
            "TED-LIUM v3",
            len(sorted_files),
            unit="speakers"
        )
    )

print("Done, thanks for playing...")