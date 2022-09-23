import argparse
import numpy as np
from vocoder.libwavernn.inference import Vocoder
import synthesizer.audio as syn_audio
from config.hparams import sp
from vocoder.models import base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick runner to test mel spectogram representation in Griffin-lim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Process the arguments
    parser.add_argument("model_fpath", type=str, help="Path to the model")
    parser.add_argument("mel_path", type=str, help= \
        "Path of the input mel spectogram file")
    parser.add_argument("wav_out", type=str, help= \
        "Path of the out wav file")
    parser.add_argument("--default_model_type", type=str, default=base.MODEL_TYPE_FATCHORD, help="default model type")
    args = parser.parse_args()

    # Setup Vocoder
    voc = Vocoder(model_fpath=args.model_fpath, model_type=args.default_model_type, verbose=True)
    voc.load(max_threads=1)  # Set to None to use all available cores

    # Encode it using LibWaveRNN
    mel = np.load(args.mel_path)
    wav = voc.vocode_mel(mel=mel)

    syn_audio.save_wav(wav, args.wav_out, sr=sp.sample_rate)
    