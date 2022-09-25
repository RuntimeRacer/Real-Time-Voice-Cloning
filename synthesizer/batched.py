import torch
from pathlib import Path
from tqdm import tqdm
from synthesizer.models import base
from synthesizer.models.tacotron import Tacotron

_model = None # type: Tacotron
_device = None # type: torch.device


def load_tacotron_model(weights_fpath: Path, device=None, use_tqdm=False):
    global _model, _device

    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = device

    _model = base.init_syn_model('tacotron', _device)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()

    if use_tqdm:
        tqdm.write("Loaded synthesizer \"%s\" trained to step %d" % (weights_fpath.name, _model.get_step()))
    else:
        print("Loaded synthesizer \"%s\" trained to step %d" % (weights_fpath.name, _model.get_step()))

def is_loaded():
    return _model is not None

def get_attention_batch(text, mel, embed):
    # Move tensors to device
    text = text.to(_device)
    mel = mel.to(_device)
    embed = embed.to(_device)

    # Forward pass to generate attention data
    with torch.no_grad():
        _, _, att_batch, _ = _model.forward(text, mel, embed)
    return att_batch.cpu()
