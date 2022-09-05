import argparse

from vocoder.train import *
from utils.argutils import print_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applies an embedded pruning mask to a pre-pruned vocoder model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments
    parser.add_argument("model_fpath", type=str, help="Path to the model")
    parser.add_argument("--default_model_type", type=str, default=base.MODEL_TYPE_FATCHORD, help="default model type")
    parser.add_argument("--out_dir", type=str, default="vocoder/pruned_models/", help="Path to the output file")
    args = parser.parse_args()

    # Process the arguments
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(exist_ok=True)

    # Run the conversion
    print_args(args, parser)
    apply_prune(**vars(args))