import torch

from vocoder.audio import *
import synthesizer.audio as syn_audio
import matplotlib.pyplot as plt
from config.hparams import sp

import soundfile as sf
import os

def gen_testset_wavernn(model, test_set, save_path, vocoder_hparams):
    step = model.get_step()

    for i, (m, x, idx) in enumerate(test_set, 1):
        if i > vocoder_hparams.gen_at_checkpoint:
            break

        print('\n| Generating: %i/%i' % (i, vocoder_hparams.gen_at_checkpoint))

        x = x[0].numpy()

        bits = 16 if vocoder_hparams.mode == 'MOL' else vocoder_hparams.bits

        if vocoder_hparams.mu_law and vocoder_hparams.mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%d_steps_%d_target.wav" % (step, i)))

        glim_mel = m[0].numpy().astype(np.float32)
        glim_wav = syn_audio.inv_mel_spectrogram(glim_mel)
        glim_wav / np.abs(glim_wav).max() * 0.97
        glim_str = "gen_griffinlim"
        glim_save_str = save_path.joinpath("%d_steps_%d_%s.wav" % (step, i, glim_str))
        syn_audio.save_wav(glim_wav, str(glim_save_str), sr=sp.sample_rate)
        
        batch_str = "gen_batched_target%d_overlap%d" % (vocoder_hparams.gen_target, vocoder_hparams.gen_overlap) if vocoder_hparams.gen_batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%d_steps_%d_%s.wav" % (step, i, batch_str))

        wav = model.generate(m, vocoder_hparams.gen_batched, vocoder_hparams.gen_target, vocoder_hparams.gen_overlap, vocoder_hparams.mu_law, sp.preemphasize)
        save_wav(wav, save_str)


@torch.no_grad()
def gen_testset_melgan(accelerator, device, step, model, criterion, test_set, save_path, vocoder_hparams):
    # change mode
    for key in model.keys():
        model[key].eval()

    # Generate from test dataset
    for i, (x_batch, y_batch) in enumerate(test_set, 1):
        if i > vocoder_hparams.gen_at_checkpoint:
            break
        print('\n| Generating: %i/%i' % (i, vocoder_hparams.gen_at_checkpoint))

        # Unwrap Models
        generator = accelerator.unwrap_model(model["generator"])
        # discriminator = accelerator.unwrap_model(model["discriminator"])

        # Generate
        x_batch = tuple([x_.to(device) for x_ in x_batch])
        y_batch = y_batch.to(device)
        y_batch_ = generator(*x_batch)
        if vocoder_hparams.generator_out_channels > 1:
            y_batch_ = criterion["pqmf"].synthesis(y_batch_)

        save_str = save_path.joinpath("steps_%d_" % (step))

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(save_str, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {step} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                sp.sample_rate,
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                sp.sample_rate,
                "PCM_16",
            )

    # restore mode
    for key in model.keys():
        model[key].train()