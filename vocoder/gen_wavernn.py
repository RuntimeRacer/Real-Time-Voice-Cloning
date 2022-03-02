from vocoder.models.fatchord_version import  WaveRNN
from vocoder.audio import *
import json
from pathlib import Path

def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):
    step = model.get_step()

    for i, (m, x, wav_path, utterance) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%d_steps_%d_target.wav" % (step, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%d_steps_%d_%s.wav" % (step, i, batch_str))

        wav = model.generate(m, batched, target, overlap, hp.mu_law)
        save_wav(wav, save_str)

def export_batch(batch, save_path):
    # Creapte path if not existing
    save_path.mkdir(parents=True, exist_ok=True)

    # Unwrap batch
    (src_mel_data, src_wav_data, src_utterance_data) = batch

    batch_meta = {}
    for i, wav in enumerate(src_mel_data):

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL':
            wav = decode_mu_law(wav, 2 ** bits, from_labels=True)
        else:
            wav = label_2_float(wav, bits)

        filename = "{}.wav".format(Path(src_wav_data[i]).with_suffix("").name)
        file_path = save_path.joinpath(filename)
        batch_meta[filename] = src_utterance_data[i]
        save_wav(wav, file_path)

    meta_fpath = Path(save_path).joinpath("batch.json")
    with meta_fpath.open("w", encoding="utf-8") as meta_file:
        json.dump(batch_meta, meta_file)