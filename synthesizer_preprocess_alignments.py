import argparse
from pathlib import Path

from synthesizer.preprocess import create_align_features
from utils.argutils import print_args

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
    parser.add_argument("-s", "--synthesizer_model_fpath", type=Path,
                        default="synthesizer/saved_models/pretrained.pt", help= \
        "Path your trained synthesizer model.")
    parser.add_argument("--skip_existing", action="store_true", default=True, help= \
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes per GPU. A synthesizer is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    create_align_features(**vars(args))
