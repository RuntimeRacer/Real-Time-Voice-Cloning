from synthesizer.preprocess import create_align_features
from utils.argutils import print_args
from pathlib import Path
import argparse

# INFO
# Only needed if intended to train forward-tactotron or fastpitch.
# This requires a default tacotron trained down to reduction factor 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=\
            "Calculate alignment scores and for the synthesizer based on output of the .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios, the embeds and the train.json file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-t", "--tacotron_model_fpath", type=Path,
                        default="synthesizer/saved_models/pretrained.pt", help= \
        "Path your trained tacotron model.")
    parser.add_argument("-t", "--threads", type=int, default=4, help= \
        "Number of threads assigned to each dataloader. Multiplies by the amount of accelerator threads.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    create_align_features(**vars(args))
