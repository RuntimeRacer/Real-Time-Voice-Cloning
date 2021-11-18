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

slr_datasets_wav = {
    "slr41": ["slr41/speakers"],
    "slr42": ["slr42/speakers"],
    "slr43": ["slr43/speakers"],
    "slr44": ["slr44/speakers"],
    "slr51": ["TEDLIUM_release-3/speakers"], # TED-LIUM v3
    "slr61": ["slr61/speakers"],
    "slr63": ["slr63/speakers"],
    "slr64": ["slr64/speakers"],
    "slr65": ["slr65/speakers"],
    "slr66": ["slr66/speakers"],
    "slr69": ["slr69/speakers"],
    "slr70": ["slr70/speakers"],
    "slr71": ["slr71/speakers"],
    "slr72": ["slr72/speakers"],
    "slr73": ["slr73/speakers"],
    "slr74": ["slr74/speakers"],
    "slr75": ["slr75/speakers"],
    "slr76": ["slr76/speakers"],
    "slr77": ["slr77/speakers"],
    "slr78": ["slr78/speakers"],
    "slr79": ["slr79/speakers"],
    "slr80": ["slr80/speakers"],
    "slr96": ["slr96/train/audio"]
}

slr_datasets_flac = {
    "slr82": ["slr82/CN-Celeb_flac/data", "slr82/CN-Celeb2_flac/data"],
    "slr100": [ # Multilingual TEDx (without translations)
        "mtedx/ar-ar/data/train",
        "mtedx/de-de/data/train",
        "mtedx/el-el/data/train",
        "mtedx/es-es/data/train",
        "mtedx/fr-fr/data/train",
        "mtedx/it-it/data/train",
        "mtedx/pt-pt/data/train",
        "mtedx/ru-ru/data/train"
    ]
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
    "VCTK": ["VCTK-Corpus/wav48_silence_trimmed"],
    "nasjonalbank": ["nasjonal-bank/speakers"]
}

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
