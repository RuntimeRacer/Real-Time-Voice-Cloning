# Readme for pre-pre-processing the different datasets for encoder training

This readme is intended to provide guidance for preprocessing datasets NOT in valid format for use with this repo.

# Encoder pre-pre processing
The encoder needs basically just audio files in the format `speaker_id/utterance_group_id/utterances`.
This format is valid for the following Datasets out of the box:

- LibriSpeech (SLR12)
- VoxCeleb 1 & 2
- LibriTTS (SLR60)
- Russian LibriSpeech (SLR96)

For other Datasets you need additional pre-processing to allow for their usage in encoder training. Here I'll share what to do for those that I have been used for encoder training.

Hints on thread counts in the following commands:
- Since I was handling the biggest portion of the data on a HDD, copy operations with more than 8 threads tended to block each other, so I kept the amount of threads low on scripts which are just copying.
- However, when there was some file format conversion involved, the 32 cores of the Ryzen Threadripper have been put to good use, since the I/O operations are only a fraction of what's actually happening there.

Hints on the min (12) and max (40) params used in the following params:
- They seemed like a good boundary to ensure speaker equality towards the later trained model.
- Minimum batch size for training I wanted to use was 10 Utterances.
- However, this reduced the size of the speaker pool quite significantly. I cannot tell yet whether I need to adapt this at a later point in time.

## Mozilla CommonVoice
Available here: https://commonvoice.mozilla.org/en/datasets

Used Version: 7.0; November 2021

Applicable versions: Down to at least 2.0

### What I did
I mostly re-used the script leftover by @sberryman; which was built around CV Version 2. However; the structure of the Data did not change since then and is still applicable for CV 7, which is the current version of Mozilla's CV Datasets. I Added a bunch of additional Arguments for the preprocessing; especially one to ignore languages. Since I tried to create an encoder which is capable of encoding ANY voice independent of language, the encoder pre-processing should not bother which language is being used (if not explicitly stated).

*Processing all languages at 16 kHz and store converted speaker files into separate target dir, using 32 threads.*
```
python scripts/commonvoice_speakers.py <source-dir>/datasets/cv-corpus-7.0-2021-07-21/ -o <target-dir>/cv-corpus-7.0-2021-07-21/ -ar 16000 -t 32
```

*Processing language "en" at 16 kHz and store converted speaker files into separate target dir, using 32 threads.*
```
python scripts/commonvoice_speakers.py <source-dir>/datasets/cv-corpus-7.0-2021-07-21/ -o <target-dir>/cv-corpus-7.0-2021-07-21/ -ar 16000 -t 32 --lang en
```

## VCTK-Corpus
Available here: https://datashare.ed.ac.uk/handle/10283/3443

Used Version: 0.92; November 2021

Applicable versions: N/A

### What I did
I created a way less complex file copy script based on the CommonVoice pre-pre-processing one, which copies over the `_mic1` recordings into the target dir and limits the amount of files copied. Since there are only 110 Speakers in this dataset, this greatly reduced the occupied disk space for this dataset. Also there is no file format conversion involved, since the files are already in compressed wav (.flac) format.

*Moving over up to 40 files for each speaker to the target directory*
```
python scripts/vctk.py /media/dominik/Project-M1nk/datasets/VCTK-Corpus/wav48_silence_trimmed/ -o /media/dominik/Project-M1nk/datasets-eval/VCTK-Corpus/wav48_silence_trimmed/ -t 8 --min 12 --max 40
```

## SLR82 (CN-Celeb 1 & 2) Datasets
Available here: http://www.openslr.org/82/

Used Version: 1 & 2; November 2021

Applicable versions: 1 & 2

### What I did
This is using almost the same pre-pre-processing scrip as `VCTK`; main focus here is to reduce the amount of files that are being used per speaker to reduce filesystem storage requirements for encoder training files. 

*CN-Celeb 1: Moving over up to 40 files for each speaker to the target directory*
```
python scripts/slr82_speakers.py /media/dominik/Project-M1nk/datasets/slr82/CN-Celeb_flac/data/ -o /media/dominik/Project-M1nk/datasets-eval/slr82/CN-Celeb_flac/data/ -t 8 --min 12
```

*CN-Celeb 2: Moving over up to 40 files for each speaker to the target directory*
```
python scripts/slr82_speakers.py /media/dominik/Project-M1nk/datasets/slr82/CN-Celeb2_flac/data/ -o /media/dominik/Project-M1nk/datasets-eval/slr82/CN-Celeb2_flac/data/ -t 8 --min 12
```


## Various generic SLR TTS Datasets
Available here: http://www.openslr.org/resources.php

Used Version: N/A; November 2021

Applicable versions: 41, 42, 43, 44, 61, 63, 64, 65, 66, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80

### What I did
Script file based on the VCTK-Corpus one. Since these SLR-Datasets so not group the speakers by folders, it is required to detect the speaker ID from the filename. This script puts them in per-speaker folders in the target directory though, so the encoder preprocessor can distinguish them.

```
# SLR41
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr41/ -o /media/dominik/Project-M1nk/datasets-eval/slr41/speakers/ --min 12 -t 8

# SLR42
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr42/ -o /media/dominik/Project-M1nk/datasets-eval/slr42/speakers/ --min 12 -t 8

# SLR43
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr43/ -o /media/dominik/Project-M1nk/datasets-eval/slr43/speakers/ --min 12 -t 8

# SLR44
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr44/ -o /media/dominik/Project-M1nk/datasets-eval/slr44/speakers/ --min 12 -t 8

# SLR61
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr61/ -o /media/dominik/Project-M1nk/datasets-eval/slr61/speakers/ --min 12 -t 8

# SLR63
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr63/ -o /media/dominik/Project-M1nk/datasets-eval/slr63/speakers/ --min 12 -t 8

# SLR64
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr64/ -o /media/dominik/Project-M1nk/datasets-eval/slr64/speakers/ --min 12 -t 8

# SLR65
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr65/ -o /media/dominik/Project-M1nk/datasets-eval/slr65/speakers/ --min 12 -t 8

# SLR66
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr66/ -o /media/dominik/Project-M1nk/datasets-eval/slr66/speakers/ --min 12 -t 8

# SLR69
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr69/ -o /media/dominik/Project-M1nk/datasets-eval/slr69/speakers/ --min 12 -t 8

# SLR70
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr70/ -o /media/dominik/Project-M1nk/datasets-eval/slr70/speakers/ --min 12 -t 8

# SLR71
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr71/ -o /media/dominik/Project-M1nk/datasets-eval/slr71/speakers/ --min 12 -t 8

# SLR72
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr72/ -o /media/dominik/Project-M1nk/datasets-eval/slr72/speakers/ --min 12 -t 8

# SLR73
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr73/ -o /media/dominik/Project-M1nk/datasets-eval/slr73/speakers/ --min 12 -t 8

# SLR74
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr74/ -o /media/dominik/Project-M1nk/datasets-eval/slr74/speakers/ --min 12 -t 8

# SLR75
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr75/ -o /media/dominik/Project-M1nk/datasets-eval/slr75/speakers/ --min 12 -t 8

# SLR76
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr76/ -o /media/dominik/Project-M1nk/datasets-eval/slr76/speakers/ --min 12 -t 8

# SLR77
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr77/ -o /media/dominik/Project-M1nk/datasets-eval/slr77/speakers/ --min 12 -t 8

# SLR78
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr78/ -o /media/dominik/Project-M1nk/datasets-eval/slr78/speakers/ --min 12 -t 8

# SLR79
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr79/ -o /media/dominik/Project-M1nk/datasets-eval/slr79/speakers/ --min 12 -t 8

# SLR80
python scripts/slr_speakers.py /media/dominik/Project-M1nk/datasets/slr80/ -o /media/dominik/Project-M1nk/datasets-eval/slr80/speakers/ --min 12 -t 8
```

## SLR 51 (TED-LIUM Release 3) Dataset
Available here: http://www.openslr.org/51/

Used Version: 3; November 2021

Applicable versions: 3

### What I did
Building on top of a pre-existing script from Sberryman, I added additional arguments to the speaker conversion script and changed it so the Talks are being converted from `.sph` format to `.wav` file format properly. The Splitting based on the STM files has just been kept mainly as it was implemented already. However, I changed the code so it now has multithreading capability.

*Create up to 40 sub-utterances without transcript files for each speaker within the target directory. Skip Speakers with less than 12 utterances*
```
python scripts/tedlium_speakers.py /media/dominik/Project-M1nk/datasets/TEDLIUM_release-3/data/ -o /media/dominik/Project-M1nk/datasets-eval/TEDLIUM_release-3/speakers -t 8 --min 12 --max 40
```

## Nasjonalbank Dataset
Available here:
- https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-16&lang=en
- https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-13&lang=en
- https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en

Used Version: N/A; November 2021

Applicable versions: N/A

### What I did
Since the existing script was already very similar to the CommonVoice script in structure, I basically just extended it so it is capable of using a threadpool while processing. Also, I discovered a bug when fetching files across all datasets, resulting in multiple speaker files ending up in the same speaker directory. I fixed it by extending the speaker ID by the name of the folder one more level above then it was, and after that I was not able to find any collisions anymore. I don't know 100% whether it's fixed now, but speaker pool increased from 2215 to 2741 after the fix, so I would assume if there are any collisions, their statistical impact is negligible.

*Moving over up to 40 files for each speaker to the target directory*
```
python scripts/nasjonal_speakers.py /media/dominik/Project-M1nk/datasets/nasjonal-bank/ -o /media/dominik/Project-M1nk/datasets-eval/nasjonal-bank/ -t 8 --min 12
```

## SLR-100 (Multilingual TEDx) Dataset
Available here: http://www.openslr.org/100/

Used Version: N/A; November 2021

Applicable versions: N/A

### What I did
Data structures seemed similar to SLR-51 (Transcription files with timestamps and one big audio file per talk), so I copied the `tedlium_speakers`-script and made necessary adaptions to run using webVTT to gather splitting times and properly create files for encoder training.

*Moving over up to 40 files for each speaker to the target directory*
```
# ar-ar
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/ar-ar/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/ar-ar/data/train -t 8 --min 12

# de-de
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/de-de/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/de-de/data/train -t 8 --min 12

# el-el
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/el-el/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/el-el/data/train -t 8 --min 12

# es-es
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/es-es/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/es-es/data/train -t 8 --min 12

# fr-fr
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/fr-fr/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/fr-fr/data/train -t 8 --min 12

# it-it
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/it-it/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/it-it/data/train -t 8 --min 12

# pt-pt
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/pt-pt/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/pt-pt/data/train -t 8 --min 12

# ru-ru
python scripts/tedx_speakers.py /media/dominik/Project-M1nk/datasets/mtedx/ru-ru/data/train -o /media/dominik/Project-M1nk/datasets-eval/mtedx/ru-ru/data/train -t 8 --min 12
```


## Copyright notes

This readme is brought to you by (c) 2021 @RuntimeRacer

Scripts described may be originally forked from @sberryman's WIP branch: https://github.com/sberryman/Real-Time-Voice-Cloning/tree/wip