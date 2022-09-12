import argparse
import numpy as np
import WaveRNNVocoder
import torch

import synthesizer.audio as syn_audio

from scipy.io.wavfile import write

from config.hparams import sp

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
    args = parser.parse_args()

    # Setup Vocoder
    vocoder = WaveRNNVocoder.Vocoder()
    vocoder.loadWeights(args.model_fpath)

    # Load the mel spectrogram and adjust its range to [-1, 1]
    mel = np.load(args.mel_path).T.astype(np.float32) / sp.max_abs_value


    # mel = np.load(args.mel_path).T.astype(np.float32) / sp.max_abs_value
    # mel = torch.as_tensor([mel])
    #
    # mel = np.concatenate(mel.numpy(), axis=1)


    wav = vocoder.melToWav(mel)
    #wav = wav.T

    syn_audio.save_wav(wav, args.wav_out, sr=sp.sample_rate)
    #write(args.wav_out, 16000, wav)
    