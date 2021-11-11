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




## Copyright notes

This readme is brought to you by (c) 2021 @RuntimeRacer

Script described may be originally forked from @sberryman's WIP branch: https://github.com/sberryman/Real-Time-Voice-Cloning/tree/wip