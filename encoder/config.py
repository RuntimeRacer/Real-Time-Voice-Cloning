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
        "clean": ["slr60/train-clean-100", "slr60/train-clean-360"],
        "other": ["slr60/train-other-500"]
    },
    "test": {
        "clean": ["slr60/test-clean"],
        "other": ["slr60/test-other"]
    },
    "dev": {
        "clean": ["slr60/dev-clean"],
        "other": ["slr60/dev-other"]
    },
}
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["voxceleb/VoxCeleb1/dev/wav"],
        "test": ["voxceleb/VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["voxceleb/VoxCeleb2/dev/wav"],
        "test": ["voxceleb/VoxCeleb2/test_wav"]
    }
}

slr_datasets = {
    "slr41": ["slr41/jv_id_male/wavs", "slr41/jv_id_female/wavs"],
    "slr42": ["slr42/km_kh_male/wavs"],
    "slr43": ["slr43/ne_np_female/wavs"],
    "slr44": ["slr44/su_id_male/wavs", "slr44/su_id_female/wavs"],
    "slr51": [], # TED-LIUM v3
    "slr61": ["slr61/es_ar_male", "slr61/es_ar_female"],
    "slr63": ["slr63/ml_in_male", "slr63/ml_in_female"],
    "slr64": ["slr64/mr_in_female"],
    "slr65": ["slr65/ta_in_male", "slr65/ta_in_female"],
    "slr66": ["slr66/te_in_male", "slr66/te_in_female"],
    "slr69": ["slr69/ca_es_male", "slr69/ca_es_female"],
    "slr70": ["slr70/en_ng_male", "slr70/en_ng_female"],
    "slr71": ["slr71/es_cl_male", "slr71/es_cl_female"],
    "slr72": ["slr72/es_co_male", "slr72/es_co_female"],
    "slr73": ["slr73/es_pe_male", "slr73/es_pe_female"],
    "slr74": ["slr74/es_pr_female"],
    "slr75": ["slr75/es_ve_male", "slr75/es_ve_female"],
    "slr76": ["slr76/eu_es_male", "slr76/eu_es_female"],
    "slr77": ["slr77/gl_es_male", "slr77/gl_es_female"],
    "slr78": ["slr78/gu_in_male", "slr78/gu_in_female"],
    "slr79": ["slr79/kn_in_male", "slr79/kn_in_female"],
    "slr80": ["slr80/my_mm_female"],
    "slr82": ["slr82/CN-Celeb/data"],
    "slr96": [],
    "slr100": []
}

commonvoice_datasets = {
    "commonvoice-7": {
        "all": ["cv-corpus-7.0-2021-07-21/speakers"],
        "en": ["cv-corpus-7.0-2021-07-21/en/speakers"]
        # TODO: other ndividual languages
    },
}

other_datasets = {
    "LJSpeech-1.1": [],
    "VCTK": ["VCTK-Corpus/wav48"],
}

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
