import ast
import pprint


class PathParams(object):
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

# This just holds path objects; mainly used for preprocessing

# Encoder
encoder = PathParams(

)

# Synthesizer
synthesizer = PathParams(
    wav_dir='wav',
    mel_dir='mels',
    embed_dir='embeds',
    duration_dir='duration',
    attention_dir='attention',
    alignment_dir='alignment',
    phoneme_pitch_dir='phoneme_pitch',
    phoneme_energy_dir='phoneme_energy'
)

# Vocoder
vocoder = PathParams(

)
