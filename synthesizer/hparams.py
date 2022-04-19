import ast
import pprint


class HParams(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

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


# Parameters for Tacotron model
hparams_tacotron = HParams(
    # Signal Processing (used in both synthesizer and vocoder)
    sample_rate=16000,
    n_fft=1024,
    num_mels=80,
    hop_size=200,      # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
    win_size=800,      # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
    fmin=40,
    fmax=8000,         # Should not exceed (sample_rate // 2)
    min_level_db=-100,
    ref_level_db=20,
    max_abs_value=4.,  # Gradient explodes if too big, premature convergence if too small.
    preemphasis=0.97,  # Filter coefficient to use if preemphasize is True
    preemphasize=True,

    # Tacotron Text-to-Speech (TTS)
    tts_embed_dims=256,  # Embedding dimension for the graphemes/phoneme inputs
    tts_encoder_dims=128,
    tts_decoder_dims=256,
    tts_postnet_dims=128,
    tts_encoder_K=16,
    tts_lstm_dims=512,
    tts_postnet_K=8,
    tts_num_highways=4,
    tts_dropout=0.5,
    tts_cleaner_names=["english_cleaners"],
    tts_stop_threshold=-3.4,  # Value below which audio generation ends.
    # For example, for a range of [-4, 4], this
    # will terminate the sequence at the first
    # frame that has all values < -3.4

    # Tacotron Training - Progressive epoch training schedule
    #
    # (r, loops, batch_size, init_lr, end_lr)
    #
    # r          = reduction factor - divisor of mel frames synthesized for each decoder iteration
    #              (lesser value => higher resolution and more precise training, but also higher step time)
    # loops      = iteration loops over whole dataset
    # batch_size = amount of dataset items to train on per step
    #
    # Learning rate is applying Stochastic Gradient Descent with Restarts (SGDR)
    # (https://markkhoffmann.medium.com/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e)
    # init_lr    = learning rate at the begin of the epoch
    # end_lr     = learning rate at the end of the epoch

    tts_schedule=[(7, 1, 112, 1e-3, 1e-6),
                  (6, 2, 100, 1e-3, 1e-6),
                  (5, 4,  88, 5e-4, 5e-7),
                  (4, 8,  76, 2e-4, 5e-7),
                  (3, 16, 64, 1e-4, 1e-7),
                  (2, 16, 44, 5e-5, 5e-8),
                  (1, 16, 22, 1e-5, 1e-8)],

    tts_clip_grad_norm=1.0,  # clips the gradient norm to prevent explosion - set to None if not needed
    tts_eval_interval=500,  # Number of steps between model evaluation (sample generation)
    # Set to -1 to generate after completing epoch, or 0 to disable

    tts_eval_num_samples=1,  # Makes this number of samples

    # Data Preprocessing
    max_mel_frames=1200,
    rescale=True,
    rescaling_max=0.9,
    synthesis_batch_size=24,
    # For vocoder preprocessing and inference. - Rule of Thumb: 1 unit per GB of VRAM of smallest card

    # Mel Visualization and Griffin-Lim
    signal_normalization=True,
    power=1.5,
    griffin_lim_iters=80,

    # Audio processing options
    allow_clipping_in_normalization=True,  # Used when signal_normalization = True
    clip_mels_length=True,  # If true, discards samples exceeding max_mel_frames
    use_lws=False,  # "Fast spectrogram phase recovery using local weighted sums"
    symmetric_mels=True,  # Sets mel range to [-max_abs_value, max_abs_value] if True, and [0, max_abs_value] if False
    trim_silence=True,  # Use with sample_rate of 16000 for best results

    # SV2TTS
    speaker_embedding_size=768,  # Dimension for the speaker embedding
    silence_min_duration_split=0.4,  # Duration in seconds of a silence for an utterance to be split
    utterance_min_duration=0.6,  # Duration in seconds below which utterances are discarded
)


# Parameters for ForwardTacotron model
hparams_forward_tacotron = HParams(

)


def hparams_debug_string():
    return str(hparams_tacotron)
