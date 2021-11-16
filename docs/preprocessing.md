# Readme for pre-pre-processing the different datasets

This readme is intended to provide guidance for preprocessing datasets NOT in valid format for use with this repo.

# Encoder pre-pre processing
The encoder needs basically just audio files in the format `speaker_id/utterance_group_id/utterances`.
This format is valid for the following Datasets out of the box:

- LibriSpeech
- VoxCeleb 1 & 2
- LibriTTS

For other Datasets you may need additional pre-processing. Here I'll share what to do for those that I have been used for encoder training:

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

*Processing languages "en" at 16 kHz and store converted speaker files into separate target dir, using 32 threads.*
```
python scripts/commonvoice_speakers.py <source-dir>/datasets/cv-corpus-7.0-2021-07-21/ -o <target-dir>/cv-corpus-7.0-2021-07-21/ -ar 16000 -t 32 --lang en
```

*Processing languages "en" at 16 kHz and store converted speaker files into "speakers" subdir within "en" source dir, using 32 threads.*
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
python scripts/vctk_speakers.py /media/dominik/Project-M1nk/datasets/VCTK-Corpus/wav48_silence_trimmed/ -o /media/dominik/Project-M1nk/datasets-eval/VCTK-Corpus/wav48_silence_trimmed/ -t 32 --min 12
```


## Copyright notes

This readme is brought to you by (c) 2021 @RuntimeRacer

Script described may be originally forked from @sberryman's WIP branch: https://github.com/sberryman/Real-Time-Voice-Cloning/tree/wip