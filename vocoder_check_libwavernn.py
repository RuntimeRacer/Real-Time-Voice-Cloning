import argparse
import numpy as np
import WaveRNNVocoder

import synthesizer.audio as syn_audio
import vocoder.audio as voc_audio

from config.hparams import wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer


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

    # Define hparams
    if args.default_model_type == base.MODEL_TYPE_FATCHORD:
        hparams = wavernn_fatchord
    elif args.default_model_type == base.MODEL_TYPE_GENEING:
        hparams = wavernn_geneing
    elif args.default_model_type == base.MODEL_TYPE_RUNTIMERACER:
        hparams = wavernn_runtimeracer
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % args.default_model_type)

    # Setup Vocoder
    vocoder = WaveRNNVocoder.Vocoder()
    vocoder.loadWeights(args.model_fpath)

    # Load the mel spectrogram and adjust its range to [-1, 1]
    mel = np.load(args.mel_path).T.astype(np.float32) / sp.max_abs_value

    # Encode it using LibWaveRNN
    wav = vocoder.melToWav(mel)

    if hparams.mu_law:
        # Do MuLaw decode over the whole generated audio for optimal normalization
        wav = voc_audio.decode_mu_law(wav, 2 ** hparams.bits, False)

    syn_audio.save_wav(wav, args.wav_out, sr=sp.sample_rate)
    