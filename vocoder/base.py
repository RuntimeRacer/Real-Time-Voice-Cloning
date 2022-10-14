import numpy as np
from config.hparams import sp, wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer, multiband_melgan
from vocoder.parallel_wavegan.models.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator
from vocoder.wavernn.models.fatchord_version import WaveRNN as WaveRNNFatchord
from vocoder.wavernn.models.geneing_version import WaveRNN as WaveRNNGeneing
from vocoder.wavernn.models.runtimeracer_version import WaveRNN as WaveRNNRuntimeRacer
from vocoder.wavernn.pruner import Pruner

# Vocoder Types
VOC_TYPE_CPP = 'libwavernn'
VOC_TYPE_PYTORCH = 'pytorch'

# Vocoder Models
MODEL_TYPE_FATCHORD = 'fatchord-wavernn'
MODEL_TYPE_GENEING = 'geneing-wavernn'
MODEL_TYPE_RUNTIMERACER = 'runtimeracer-wavernn'
MODEL_TYPE_MULTIBAND_MELGAN = 'multiband-melgan'


def init_voc_model(model_type, device, override_hp_fatchord=None, override_hp_geneing=None, override_hp_runtimeracer=None):
    model = None
    pruner = None
    if model_type == MODEL_TYPE_FATCHORD:
        hparams = wavernn_fatchord
        if override_hp_fatchord is not None:
            hparams = override_hp_fatchord

        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(hparams.upsample_factors)[-1] == sp.hop_size

        model = WaveRNNFatchord(
            rnn_dims=hparams.rnn_dims,
            fc_dims=hparams.fc_dims,
            bits=hparams.bits,
            pad=hparams.pad,
            upsample_factors=hparams.upsample_factors,
            feat_dims=sp.num_mels,
            compute_dims=hparams.compute_dims,
            res_out_dims=hparams.res_out_dims,
            res_blocks=hparams.res_blocks,
            hop_length=sp.hop_size,
            sample_rate=sp.sample_rate,
            mode=hparams.mode,
            pruning=True
        ).to(device)

        # Setup pruner if enabled
        if hparams.use_sparsification:
            pruner = Pruner(hparams.start_prune, hparams.prune_steps, hparams.sparsity_target, hparams.sparse_group)
            pruner.update_layers(model.prune_layers, True)
    elif model_type == MODEL_TYPE_GENEING:
        hparams = wavernn_geneing
        if override_hp_geneing is not None:
            hparams = override_hp_geneing

        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(hparams.upsample_factors)[-1] == sp.hop_size

        model = WaveRNNGeneing(
            rnn_dims=hparams.rnn_dims,
            fc_dims=hparams.fc_dims,
            bits=hparams.bits,
            pad=hparams.pad,
            upsample_factors=hparams.upsample_factors,
            feat_dims=sp.num_mels,
            compute_dims=hparams.compute_dims,
            res_out_dims=hparams.res_out_dims,
            res_blocks=hparams.res_blocks,
            hop_length=sp.hop_size,
            sample_rate=sp.sample_rate,
            mode=hparams.mode,
            pruning=True
        ).to(device)

        # Setup pruner if enabled
        if hparams.use_sparsification:
            pruner = Pruner(hparams.start_prune, hparams.prune_steps, hparams.sparsity_target, hparams.sparse_group)
            pruner.update_layers(model.prune_layers, True)
    elif model_type == MODEL_TYPE_RUNTIMERACER:
        hparams = wavernn_runtimeracer
        if override_hp_runtimeracer is not None:
            hparams = override_hp_runtimeracer

        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(hparams.upsample_factors)[-1] == sp.hop_size

        model = WaveRNNRuntimeRacer(
            rnn_dims=hparams.rnn_dims,
            fc_dims=hparams.fc_dims,
            bits=hparams.bits,
            pad=hparams.pad,
            upsample_factors=hparams.upsample_factors,
            feat_dims=sp.num_mels,
            compute_dims=hparams.compute_dims,
            res_out_dims=hparams.res_out_dims,
            res_blocks=hparams.res_blocks,
            hop_length=sp.hop_size,
            sample_rate=sp.sample_rate,
            mode=hparams.mode,
            pruning=True
        ).to(device)

        # Setup pruner if enabled
        if hparams.use_sparsification:
            pruner = Pruner(hparams.start_prune, hparams.prune_steps, hparams.sparsity_target, hparams.sparse_group)
            pruner.update_layers(model.prune_layers, True)

    elif model_type == MODEL_TYPE_MULTIBAND_MELGAN:
        # Determine which Generator to use
        if multiband_melgan.generator_type == "MelGANGenerator":
            generator = MelGANGenerator(
                in_channels=multiband_melgan.generator_in_channels,
                out_channels=multiband_melgan.generator_out_channels,
                kernel_size=multiband_melgan.generator_kernel_size,
                channels=multiband_melgan.generator_channels,
                upsample_scales=multiband_melgan.generator_upsample_scales,
                stack_kernel_size=multiband_melgan.generator_stack_kernel_size,
                stacks=multiband_melgan.generator_stacks,
                use_weight_norm=multiband_melgan.generator_use_weight_norm,
                use_causal_conv=multiband_melgan.generator_use_causal_conv,
            ).to(device)
        else:
            raise NotImplementedError("Invalid generator of type '%s' provided. Aborting..." % multiband_melgan.generator_type)

        # Determine which Discriminator to use
        if multiband_melgan.discriminator_type == "MelGANMultiScaleDiscriminator":
            discriminator = MelGANMultiScaleDiscriminator(
                in_channels=multiband_melgan.discriminator_in_channels,
                out_channels=multiband_melgan.discriminator_out_channels,
                scales=multiband_melgan.discriminator_scales,
                downsample_pooling=multiband_melgan.discriminator_downsample_pooling,
                downsample_pooling_params=multiband_melgan.dicriminator_downsample_pooling_params,
                kernel_sizes=multiband_melgan.dicriminator_kernel_sizes,
                channels=multiband_melgan.dicriminator_channels,
                max_downsample_channels=multiband_melgan.dicriminator_max_downsample_channels,
                downsample_scales=multiband_melgan.dicriminator_downsample_scales,
                nonlinear_activation=multiband_melgan.dicriminator_nonlinear_activation,
                nonlinear_activation_params=multiband_melgan.dicriminator_nonlinear_activation_params,
                use_weight_norm=multiband_melgan.dicriminator_use_weight_norm,
            ).to(device)
        else:
            raise NotImplementedError("Invalid discriminator of type '%s' provided. Aborting..." % multiband_melgan.discriminator_type)

        # GAN Models consist of 2 models actually, so we use a dict mapping here instead.
        model = {
            "type": model_type,
            "generator": generator,
            "discriminator": discriminator
        }
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return model, pruner


def get_model_type(model):
    if isinstance(model, WaveRNNFatchord):
        return MODEL_TYPE_FATCHORD
    elif isinstance(model, WaveRNNGeneing):
        return MODEL_TYPE_GENEING
    elif isinstance(model, WaveRNNRuntimeRacer):
        return MODEL_TYPE_RUNTIMERACER
    elif isinstance(model, dict) and "type" in model:
        # For composite models
        return model["type"]
    else:
        raise NotImplementedError("Provided object is not a valid vocoder model.")
