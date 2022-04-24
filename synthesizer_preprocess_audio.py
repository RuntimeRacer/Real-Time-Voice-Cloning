from synthesizer.preprocess import synthesizer_preprocess_dataset
from synthesizer import config
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your synthesizer training datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=None, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", default=True, help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--no_alignments", action="store_true", default=True, help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)")
    parser.add_argument("-d", "--datasets", type=str,
                        default="VCTK-Corpus,cv-corpus-7.0-2021-07-21,LibriTTS,TEDLIUM_release-3", help=\
        "Comma-separated list of the name of the datasets you want to preprocess.")
    #parser.add_argument("--subfolders", type=str, default="train-clean-100, train-clean-360", help=\
    #    "Comma-separated list of subfolders to process inside your dataset directory")
    args = parser.parse_args()

    # Verify webrtcvad is available
    try:
        import webrtcvad
    except:
        raise ModuleNotFoundError("Package 'webrtcvad' not found. This package enables "
            "noise removal and is recommended. Please install and try again. If installation fails, "
            "use --no_trim to disable this error message.")

    # Process the arguments
    # List of datasets
    args.datasets = args.datasets.split(",")

    # Output dir for processed audio files
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Print the arguments
    print_args(args, parser)

    # Convert args object to param dict for function
    args = vars(args)

    # Process each dataset
    for dataset in args.pop("datasets"):
        # Fill in individual args
        args["dataset_name"] = dataset
        args["subfolders"] = config.datasets[dataset]["directories"]
        args["audio_extensions"] = config.datasets[dataset]["audio_extensions"]
        args["transcript_extension"] = config.datasets[dataset]["transcript_extension"]
        # Execute preprocess
        synthesizer_preprocess_dataset(**args)
