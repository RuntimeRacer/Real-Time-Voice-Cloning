import torch
from synthesizer import audio
from config.hparams import tacotron as hp_tacotron, sp, preprocessing
from synthesizer.models import base
from synthesizer.utils.text import text_to_sequence
from vocoder.display import simple_table
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa

# Not 100% sure if required for proper threading, but seems META to quickly fix slow alignment generation
_model = None # type: Synthesizer
_device = None # type: torch.device

def load_model(weights_fpath: Path, model_type='tacotron'):
    global _model, _device
    _model = Synthesizer(model_type=model_type, model_fpath=weights_fpath)
    _model.load()

def is_loaded():
    return _model is not None

def synthesize_spectrogram(text, embedding, return_alignments=True):
    texts = [text]
    embeds = [embedding] * len(texts)
    return _model.synthesize_spectrograms(texts, embeds, return_alignments)


class Synthesizer:
    
    def __init__(
            self,
            model_type: str,
            model_fpath: Path,
            verbose=True):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.

        :param model_type: type of the synthesizer model, must be one of: 'tacotron', 'forward-tacotron', 'fastpitch'
        :param model_fpath: path to the trained model file
        :param hparams: HParams object containing all hyperparameters relevant for this synthesizer model
        :param verbose: if False, prints less information when using the model
        """
        self.model_type = model_type
        self.model_fpath = model_fpath
        self.verbose = verbose
 
        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.verbose:
            print("Synthesizer using device:", self.device)
        
        # Synthesizer model will be instantiated later on first use.
        self._model = None

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._model is not None
    
    def load(self):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        """
        try:
            self._model = base.init_syn_model(self.model_type, self.device)
        except NotImplementedError as e:
            print(str(e))
            return

        self._model.load(self.model_fpath)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer of model '%s' at path '%s'." % (self.model_type, self.model_fpath.name))
            print("Model has been trained to step %d." % (self._model.state_dict()["step"]))

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the 
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Load the model on the first request.
        if not self.is_loaded():
            self.load()

            # Print some info about the model when it is loaded            
            tts_k = self._model.get_step() // 1000

            simple_table([("Tacotron", str(tts_k) + "k"),
                        ("r", self._model.r)])

        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip(), preprocessing.cleaner_names) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i + preprocessing.synthesis_batch_size]
                          for i in range(0, len(inputs), preprocessing.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i + preprocessing.synthesis_batch_size]
                          for i in range(0, len(embeddings), preprocessing.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            _, mels, alignments = self._model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < hp_tacotron.stop_threshold:
                    m = m[:, :-1]
                specs.append(m)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer. 
        """
        wav = librosa.load(str(fpath), sp.sample_rate)[0]
        if preprocessing.rescale:
            wav = wav / np.abs(wav).max() * preprocessing.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that 
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav
        
        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
        return mel_spectrogram
    
    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel)


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
