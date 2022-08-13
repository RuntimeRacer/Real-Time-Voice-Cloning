from vocoder.audio import *
import json
from pathlib import Path

def analyse_and_export_batch(batch, dataset, individual_loss, save_path, vocoder_hparams):
    # Creapte path if not existing
    save_path.mkdir(parents=True, exist_ok=True)

    # Unwrap batch
    (src_mel_data, src_wav_data, indices) = batch

    # Iterate through loss values and gather metadata
    batch_meta = []
    for i, loss in enumerate(individual_loss):
        meta_idx = indices[i]

        (mel_path, wav_path) = dataset.samples_fpaths[meta_idx]
        text = dataset.samples_texts[meta_idx]
        #syn_mel = src_mel_data[i]
        wav_mel = src_wav_data[i]

        syn_filename = "{}_syn.wav".format(wav_path.with_suffix("").name)
        wav_filename = "{}.wav".format(wav_path.with_suffix("").name)

        # Convert mels
        bits = 16 if hp_wavernn.mode == 'MOL' else hp_wavernn.bits
        if hp_wavernn.mu_law and hp_wavernn.mode != 'MOL':
            wav_mel = decode_mu_law(wav_mel, 2 ** bits, from_labels=True)
        else:
            wav_mel = label_2_float(wav_mel, bits)

        #mel_length = int(dataset.metadata[meta_idx][4])
        #syn_mel = decode_mu_law(syn_mel, 2 ** bits, from_labels=True)

        # Save wavs
        #syn_file_path = save_path.joinpath(syn_filename)
        #save_wav(syn_mel, syn_file_path)
        wav_file_path = save_path.joinpath(wav_filename)
        save_wav(wav_mel, wav_file_path)

        # Put info in list
        batch_meta.append((loss, wav_filename, text))

    # Sort list by loss descending
    batch_meta.sort(key=lambda x:x[0], reverse=True)

    meta_fpath = Path(save_path).joinpath("batch.json")
    with meta_fpath.open("w", encoding="utf-8") as meta_file:
        json.dump(batch_meta, meta_file)