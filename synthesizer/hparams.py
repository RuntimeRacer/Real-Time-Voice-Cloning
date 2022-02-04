import ast
import pprint

class HParams(object):
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

hparams = HParams(
        ### Signal Processing (used in both synthesizer and vocoder)
        sample_rate = 22050,
        n_fft = 2048,
        num_mels = 120,
        hop_size = 300,                             # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
        win_size = 1200,                            # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
        fmin = 40,
        min_level_db = -100,
        ref_level_db = 20,
        max_abs_value = 4.,                        # Gradient explodes if too big, premature convergence if too small.
        preemphasis = 0.97,                         # Filter coefficient to use if preemphasize is True
        preemphasize = True,

        ### Tacotron Text-to-Speech (TTS)
        tts_embed_dims = 256,                       # Embedding dimension for the graphemes/phoneme inputs
        tts_encoder_dims = 128,
        tts_decoder_dims = 256,
        tts_postnet_dims = 128,
        tts_encoder_K = 16,
        tts_lstm_dims = 512,
        tts_postnet_K = 8,
        tts_num_highways = 4,
        tts_dropout = 0.5,
        tts_cleaner_names = ["english_cleaners"],
        tts_stop_threshold = -3.4,                  # Value below which audio generation ends.
                                                    # For example, for a range of [-4, 4], this
                                                    # will terminate the sequence at the first
                                                    # frame that has all values < -3.4

        ### Tacotron Training
        sgdr_init_lr = 1e-3
        sgdr_final_lr = 1e-7

        # Progressive training schedule
        # (r, lr, loops, batch_size)
        # r          = reduction factor (# of mel frames synthesized for each decoder iteration)
        # lr         = learning rate
        # loops      = iteration loops over whole dataset
        # batch_size = amount of dataset items to train on per step
        #
        tts_schedule = [(7,  1e-3,  4,  80),
                        (6,  5e-4,  4,  80),
                        (5,  2e-4,  4,  80),
                        (5,  1e-4,  8,  80),
                        # After finishing 1st epoch of lr 1e-4, reduce batch size to smoothen training optimization
                        (4,  1e-4,  4,  64),
                        (3,  1e-4,  4,  48),
                        (2,  5e-5,  4,  32),
                        (2,  3e-5,  4,  32),
                        (2,  1e-5,  4,  32),
                        # Fine-tuning after finishing epoch of lr 1e-5
                        (2,  5e-6,  4,  16),
                        (2,  1e-6,  4,  16)],

        tts_clip_grad_norm = 1.0,                   # clips the gradient norm to prevent explosion - set to None if not needed
        tts_eval_interval = 500,                    # Number of steps between model evaluation (sample generation)
                                                    # Set to -1 to generate after completing epoch, or 0 to disable

        tts_eval_num_samples = 1,                   # Makes this number of samples

        ### Data Preprocessing
        max_mel_frames = 1200,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 24,                  # For vocoder preprocessing and inference. - Rule of Thumb: 1 unit per GB of VRAM of smallest card

        ### Mel Visualization and Griffin-Lim
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 80,

        ### Audio processing options
        fmax = 11000,                               # Should not exceed (sample_rate // 2)
        allow_clipping_in_normalization = True,     # Used when signal_normalization = True
        clip_mels_length = True,                    # If true, discards samples exceeding max_mel_frames
        use_lws = False,                            # "Fast spectrogram phase recovery using local weighted sums"
        symmetric_mels = True,                      # Sets mel range to [-max_abs_value, max_abs_value] if True,
                                                    #               and [0, max_abs_value] if False
        trim_silence = True,                        # Use with sample_rate of 16000 for best results

        ### SV2TTS
        speaker_embedding_size = 768,               # Dimension for the speaker embedding
        silence_min_duration_split = 0.3,           # Duration in seconds of a silence for an utterance to be split
        utterance_min_duration = 0.6,               # Duration in seconds below which utterances are discarded
        )

def hparams_debug_string():
    return str(hparams)
