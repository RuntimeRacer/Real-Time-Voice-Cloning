from config.hparams import tacotron as hp_tacotron, forward_tacotron as hp_forward_tacotron, sp, sv2tts

from synthesizer.models.tacotron import Tacotron
from synthesizer.models.forward_tacotron import ForwardTacotron

from synthesizer.utils.symbols import symbols

# Synthesizer Models
MODEL_TYPE_TACOTRON = 'tacotron'
MODEL_TYPE_FORWARD_TACOTRON = 'forward-tacotron'


def init_syn_model(model_type, device):
    model = None
    if model_type == MODEL_TYPE_TACOTRON:
        model = Tacotron(
            embed_dims=hp_tacotron.embed_dims,
            num_chars=len(symbols),
            encoder_dims=hp_tacotron.encoder_dims,
            decoder_dims=hp_tacotron.decoder_dims,
            n_mels=sp.num_mels,
            fft_bins=sp.num_mels,
            postnet_dims=hp_tacotron.postnet_dims,
            encoder_K=hp_tacotron.encoder_K,
            lstm_dims=hp_tacotron.lstm_dims,
            postnet_K=hp_tacotron.postnet_K,
            num_highways=hp_tacotron.num_highways,
            dropout=hp_tacotron.dropout,
            stop_threshold=hp_tacotron.stop_threshold,
            speaker_embedding_size=sv2tts.speaker_embedding_size
        ).to(device)
    elif model_type == MODEL_TYPE_FORWARD_TACOTRON:
        model = ForwardTacotron(
            embed_dims=hp_forward_tacotron.embed_dims,
            series_embed_dims=hp_forward_tacotron.series_embed_dims,
            num_chars=len(symbols),
            n_mels=sp.num_mels,
            durpred_conv_dims=hp_forward_tacotron.duration_conv_dims,
            durpred_rnn_dims=hp_forward_tacotron.duration_rnn_dims,
            durpred_dropout=hp_forward_tacotron.duration_dropout,
            pitch_conv_dims=hp_forward_tacotron.pitch_conv_dims,
            pitch_rnn_dims=hp_forward_tacotron.pitch_rnn_dims,
            pitch_dropout=hp_forward_tacotron.pitch_dropout,
            pitch_strength=hp_forward_tacotron.pitch_strength,
            energy_conv_dims=hp_forward_tacotron.energy_conv_dims,
            energy_rnn_dims=hp_forward_tacotron.energy_rnn_dims,
            energy_dropout=hp_forward_tacotron.energy_dropout,
            energy_strength=hp_forward_tacotron.energy_strength,
            prenet_dims=hp_forward_tacotron.prenet_dims,
            prenet_k=hp_forward_tacotron.prenet_k,
            prenet_num_highways=hp_forward_tacotron.prenet_num_highways,
            prenet_dropout=hp_forward_tacotron.prenet_dropout,
            rnn_dims=hp_forward_tacotron.rnn_dims,
            postnet_dims=hp_forward_tacotron.postnet_dims,
            postnet_k=hp_forward_tacotron.postnet_k,
            postnet_num_highways=hp_forward_tacotron.postnet_num_highways,
            postnet_dropout=hp_forward_tacotron.postnet_dropout,
            speaker_embed_dims=sv2tts.speaker_embedding_size
        ).to(device)
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return model


def get_model_train_elements(model_type):
    train_elements = []
    if model_type == MODEL_TYPE_TACOTRON:
        train_elements = ["mel", "embed"]
    elif model_type == MODEL_TYPE_FORWARD_TACOTRON:
        train_elements = ["mel", "embed", "duration", "attention", "alignment", "phoneme_pitch", "phoneme_energy"]
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)
    return train_elements


def get_model_type(model):
    if isinstance(model, Tacotron):
        return MODEL_TYPE_TACOTRON
    elif isinstance(model, ForwardTacotron):
        return MODEL_TYPE_FORWARD_TACOTRON
    else:
        raise NotImplementedError("Provided object is not a valid synthesizer model.")
