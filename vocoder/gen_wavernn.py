from vocoder.audio import *
import synthesizer.audio as syn_audio
from config.hparams import sp, preprocessing

def gen_testset(model, test_set, save_path, vocoder_hparams):
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

        # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
        # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
        if preprocessing.symmetric_mels:
            mel_pad_value = -1 * sp.max_abs_value
        else:
            mel_pad_value = 0

        glim_mel = m.T
        glim_wav = syn_audio.inv_mel_spectrogram(pad2d(glim_mel, glim_mel.shape[1], pad_value=mel_pad_value))
        glim_str = "gen_griffinlim"
        glim_save_str = save_path.joinpath("%d_steps_%d_%s.wav" % (step, i, glim_str))
        syn_audio.save_wav(glim_wav, str(glim_save_str), sr=sp.sample_rate)
        
        batch_str = "gen_batched_target%d_overlap%d" % (vocoder_hparams.gen_target, vocoder_hparams.gen_overlap) if vocoder_hparams.gen_batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%d_steps_%d_%s.wav" % (step, i, batch_str))

        wav = model.generate(m, vocoder_hparams.gen_batched, vocoder_hparams.gen_target, vocoder_hparams.gen_overlap, vocoder_hparams.mu_law, sp.preemphasize)
        save_wav(wav, save_str)


def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)