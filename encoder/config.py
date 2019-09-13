librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

slr_datasets = {
    # 39
    "slr41": ["slr41/jv_id_male/wavs", "slr41/jv_id_female/wavs"],
    # 16
    "slr42": ["slr42/km_kh_male/wavs"],
    # 18
    "slr43": ["slr43/ne_np_female/wavs"],
    # 41
    "slr44": ["slr44/su_id_male/wavs", "slr44/su_id_female/wavs"],
    # 44
    "slr61": ["slr61/es_ar_male", "slr61/es_ar_female"],
    # 42
    "slr63": ["slr63/ml_in_male", "slr63/ml_in_female"],
    # 9
    "slr64": ["slr64/mr_in_female"],
    # 50
    "slr65": ["slr65/ta_in_male", "slr65/ta_in_female"],
    # 47
    "slr66": ["slr66/te_in_male", "slr66/te_in_female"],
    # 36
    "slr69": ["slr69/ca_es_male", "slr69/ca_es_female"],
    # 31
    "slr70": ["slr70/en_ng_male", "slr70/en_ng_female"],
    # 31
    "slr71": ["slr71/es_cl_male", "slr71/es_cl_female"],
    # 33
    "slr72": ["slr72/es_co_male", "slr72/es_co_female"],
    # 38
    "slr73": ["slr73/es_pe_male", "slr73/es_pe_female"],
    # 5
    "slr74": ["slr74/es_pr_female"],
    # 23
    "slr75": ["slr75/es_ve_male", "slr75/es_ve_female"],
    # 52
    "slr76": ["slr76/eu_es_male", "slr76/eu_es_female"],
    # 44
    "slr77": ["slr77/gl_es_male", "slr77/gl_es_female"],
    # 36
    "slr78": ["slr78/gu_in_male", "slr78/gu_in_female"],
    # 59
    "slr79": ["slr79/kn_in_male", "slr79/kn_in_female"],
    # 20
    "slr80": ["slr80/my_mm_female"],
}

other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
