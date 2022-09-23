import math
import time
from concurrent.futures import ThreadPoolExecutor

from config.hparams import sp, wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer
from vocoder.models import base
from vocoder.audio import decode_mu_law, de_emphasis
from vocoder.utils import folding
from pathlib import Path
import numpy as np
import torch
import psutil
import WaveRNNVocoder


class Vocoder:

    def __init__(
            self,
            model_fpath: Path,
            model_type: str,
            verbose=True):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.
        :param model_fpath: path to the trained, binary converted model file
        :param model_type: type of the model; for proper hparam mapping
        :param verbose: if False, prints less information when using the model
        """
        self.model_fpath = model_fpath
        self.model_type = model_type
        self.verbose = verbose

        if verbose:
            print("Instantiated C++ WaveRNN Vocoder wrapper for model: ", self.model_fpath)

        # For high speed inference we need to make best utilization of CPU resources and threads.
        self._processing_thread_wrappers = None

    def load(self, max_threads=None):
        """
        Determines best processing setup for high speed C++ inference and sets everything up.
        """
        # Get max availiable physical threads of this CPU
        # TODO: Find out if this is faster when using physical vs logical
        cpus = psutil.cpu_count(logical=False)
        if max_threads is not None and max_threads < cpus:
            cpus = max_threads

        # Init wrapper list
        self._processing_thread_wrappers = []
        for tID in range(cpus):
            # Init vocoder wrapper for the model file
            vocoder = WaveRNNVocoder.Vocoder()
            vocoder.loadWeights(self.model_fpath)
            # Append it tot the wrapper list
            self._processing_thread_wrappers.append(vocoder)

    def vocode_mel(self, mel, progress_callback=None):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer.

        :param mel: The mel from the synthesizer
        :param progress_callback: Callback for notifying caller about processing progress
        :return: vocoded waveform ready to be saved
        """

        if self.model_type == base.MODEL_TYPE_FATCHORD:
            hp_wavernn = wavernn_fatchord
        elif self.model_type == base.MODEL_TYPE_GENEING:
            hp_wavernn = wavernn_geneing
        elif self.model_type == base.MODEL_TYPE_RUNTIMERACER:
            hp_wavernn = wavernn_runtimeracer
        else:
            raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % self.model_type)

        # Adjust mel range range to [-1, 1] (normalization)
        mel = mel.T.astype(np.float32) / sp.max_abs_value

        output = None
        wrapper_count = len(self._processing_thread_wrappers)
        if wrapper_count == 0:
            raise RuntimeError("No processing thread wrappers. Did you properly load the Vocoder instance? Aborting...")
        elif wrapper_count == 1:
            output = self.vocode_thread(0, mel)
        else:
            """
            Determine optimal size for folding.
            - For a small mel we might not need all avaliable cores.
            - For a larger mel which would have more batch elements than cores, it is more efficient to increase the
              amount of frames for each core so we can compute the whole wav in a single threading cycle.
            """
            min_target = hp_wavernn.gen_target
            min_overlap = hp_wavernn.gen_overlap

            _, mel_size, _ = mel.size()
            optimal_target = math.ceil(((mel_size - min_overlap) / wrapper_count) - min_overlap)
            if optimal_target < min_target:
                optimal_target = min_target

            # Do the folding
            mels = mel
            with torch.no_grad():
                wave_len = (mels.size(-1) - 1) * sp.hop_size
                mels = folding.fold_with_overlap(mels, optimal_target, min_overlap)

            # Render using multithreading
            chunks_count = len(mels)
            threads = []
            with ThreadPoolExecutor(max_workers=wrapper_count) as executor:
                for tID in range(chunks_count):
                    threads.append(executor.submit(self.vocode_thread, tID, mels[tID]))

            # wait for processing done
            processing_done = False
            while not processing_done:
                done_cnt = 0
                for t in threads:
                    if t.done():
                        done_cnt += 1
                if done_cnt == len(threads):
                    processing_done = True
                else:
                    time.sleep(0.01)

            # Unfold the results
            output = []
            for t in threads:
                output.append(t.result().numpy())

            with torch.no_grad():
                output = torch.stack(output).transpose(0, 1)
                output = output.cpu().numpy()
                output = output.astype(np.float64)
                output = folding.xfade_and_unfold(output, optimal_target, min_overlap)

        if hp_wavernn.mu_law:
            # Do MuLaw decode over the whole generated audio for optimal normalization
            output = decode_mu_law(output, 2 ** hp_wavernn.bits, False)
        if sp.preemphasize:
            output = de_emphasis(output)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * sp.hop_size)
        output = output[:wave_len]
        output[-20 * sp.hop_size:] *= fade_out

        return output

    def vocode_thread(self, tID, chunk):
        return self._processing_thread_wrappers[tID].melToWav(chunk)

