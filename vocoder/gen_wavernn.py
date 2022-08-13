from vocoder.models.fatchord_version import WaveRNN
from vocoder.audio import *
from config.hparams import sp

def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path, vocoder_hparams):
    step = model.get_step()

    for i, (m, x, idx) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if vocoder_hparams.mode == 'MOL' else vocoder_hparams.bits

        if vocoder_hparams.mu_law and vocoder_hparams.mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%d_steps_%d_target.wav" % (step, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%d_steps_%d_%s.wav" % (step, i, batch_str))

        wav = model.generate(m, batched, target, overlap, vocoder_hparams.mu_law, sp.preemphasize)
        save_wav(wav, save_str)

