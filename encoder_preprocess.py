from encoder import config
from encoder.preprocess import encoder_preprocess_dataset
from utils.argutils import print_args
from pathlib import Path
import argparse

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                    "writes them to the disk. This will allow you to train the encoder. The "
                    "datasets required are at least one of VoxCeleb1, VoxCeleb2 and LibriSpeech. "
                    "Ideally, you should have all three. You should extract them as they are "
                    "after having downloaded them and put them in a same directory, e.g.:\n"
                    "-[datasets_root]\n"
                    "  -LibriSpeech\n"
                    "    -train-other-500\n"
                    "  -VoxCeleb1\n"
                    "    -wav\n"
                    "    -vox1_meta.csv\n"
                    "  -VoxCeleb2\n"
                    "    -dev",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your encoder training datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms. If left out, "
        "defaults to <datasets_root>/SV2TTS/encoder/")
    parser.add_argument("-d", "--datasets", type=str,
                        default="libritts_other:wav,voxceleb1:wav,voxceleb2:wav", help=\
        "Comma-separated list of the name of the datasets you want to preprocess; plus the file type to be expected by the pre-processor.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    parser.add_argument('--min', type=int, default=12, help=\
        'Minimum number of files per speaker')
    parser.add_argument('--max', type=int, default=40, help=\
        'Maximum number of files per speaker')
    parser.add_argument("-t", "--threads", type=int, default=8)
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
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Print the arguments
    print_args(args, parser)

    # Helper function for dataset merge TODO: Move this out of here and into a generic function
    def merge_datasets_paths(datasets: dict):
        result = []
        for l in datasets.values():
            result += l
        return result

    # Mapping of datasets to config values. TODO: Make this more dynamic
    dataset_mapping = {
        "librispeech_other": config.librispeech_datasets["train"]["other"],
        "libritts_other": config.libritts_datasets["train"]["other"],
        "voxceleb1": config.voxceleb_datasets["voxceleb1"]["train"],
        "voxceleb2": config.voxceleb_datasets["voxceleb2"]["train"],
        "vctk": config.other_datasets["VCTK"],
        "commonvoice_all": config.commonvoice_datasets["commonvoice-7"]["all"],
        "slr_wav": merge_datasets_paths(config.slr_datasets_wav),
        "slr_100": config.slr_datasets_wav["slr100"],
        "slr_flac": merge_datasets_paths(config.slr_datasets_flac),
        "nasjonalbank": config.other_datasets["nasjonalbank"]
    }

    # Convert args object to param dict for function
    args = vars(args)

    # Process each dataset
    for dataset in args.pop("datasets"):
        # Get details
        dataset_details = dataset.split(":")
        if len(dataset_details) == 1:
            dataset_details.extend("wav")
        elif len(dataset_details) != 2:
            print("Error: Invalid data for dataset '{0}'. Aborting..." % dataset)
            exit()

        # Name and filetype
        dataset_name = dataset_details[0]
        file_type = dataset_details[1]

        # check if we have a mapping
        # add defined dataset folders as 'dataset_paths' and filetype as 'file_type':
        if dataset_name not in dataset_mapping:
            print("Error: No mapping found for dataset '{0}'. Aborting..." % dataset_name)
            exit()
        else:
            args["dataset_paths"] = dataset_mapping[dataset_name]
            args["file_type"] = file_type
        
        # Start preprocessing
        encoder_preprocess_dataset(**args)


        # preprocess_func[dataset](**args)
        # FIXME: Code for language based preprocessing, probably not needed.
        # if dataset[0:3] == "slr":
        #     args["slr_dataset"] = dataset
        #     preprocess_slr(**args)
        # elif dataset[0:2] == "cv":
        #     args["lang"] = dataset[3:]
        #     preprocess_commonvoice(**args)
        # elif dataset[0:8] == "nasjonal":
        #     args["lang"] = dataset[9:]
        #     preprocess_nasjonal(**args)
        # else:
        #     preprocess_func[dataset](**args)
