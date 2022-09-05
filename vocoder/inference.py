
from config.hparams import sp, wavernn_fatchord, wavernn_geneing
from vocoder.models import base
import torch


_model = None
_model_type = None


def load_model(weights_fpath, verbose=True):
    global _model, _device, _model_type

    if torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')

    # Load model weights from provided model path
    checkpoint = torch.load(weights_fpath, map_location=_device)
    _model_type = base.MODEL_TYPE_FATCHORD
    if "model_type" in checkpoint:
        _model_type = checkpoint["model_type"]

    # Init the model
    try:
        _model, _ = base.init_voc_model(_model_type, _device)
        _model = _model.eval()
    except NotImplementedError as e:
        print(str(e))
        return

    # Load model state
    _model.load_state_dict(checkpoint["model_state"])
    
    if verbose:
        print("Loaded synthesizer of model '%s' at path '%s'." % (_model_type, weights_fpath))
        print("Model has been trained to step %d." % (_model.state_dict()["step"]))


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True,  batched=True, target=None, overlap=None, progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None or _model_type is None:
        raise Exception("Please load Wave-RNN in memory before using it")

    if _model_type == base.MODEL_TYPE_FATCHORD:
        hp_wavernn = wavernn_fatchord
    elif _model_type == base.MODEL_TYPE_GENEING:
        hp_wavernn = wavernn_geneing
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % _model_type)

    if target is None:
        target = hp_wavernn.gen_target
    if overlap is None:
        overlap = hp_wavernn.gen_overlap

    if normalize:
        mel = mel / sp.max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp_wavernn.mu_law, sp.preemphasize, progress_callback)
    return wav
