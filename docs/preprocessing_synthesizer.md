# Readme for pre-pre-processing the different datasets for synthesizer training

This readme is intended to provide guidance for preprocessing datasets NOT in valid format for use with this repo.

# Synthesizer pre-pre processing
The synthesizer largely can use the same datasets as the encoder during training, but it also requires transcriptions for the spoken text.

There are 2 modes:
- a) Synthesize using alignments
- b) Synthesize without alignments

Format a) is valid for the following Datasets out of the box:

- LibriSpeech (SLR12)

Format b) is valid for the following Datasets out of the box:
- LibriSpeech (SLR12)
- LibriTTS (SLR60)

As far as I understood, Format a) is especially useful for training individual word utterances. For the first example I am working on here however, I'll skip alignment training and just train against large datasets which have transcripts, but no alignments.

## Mozilla CommonVoice
Available here: https://commonvoice.mozilla.org/en/datasets

Used Version: 7.0; November 2021

Applicable versions: Down to at least 2.0

### What I did
Based on the work I did for the encoder pre-pre-processing for CV-7, I merged my optimizations together with the existing script from @sberryman to build a new version of the script. I also wanted to use more variations of data for synthesizer training (using all data available in CV-7 english) than I did for the encoder training (Only speakers with 12+ utterances).

*Processing language "en" at 16 kHz and store converted speaker files into separate target dir, using 24 threads.*
```
python scripts/commonvoice_transcript.py /media/dominik/Project-M1nk/datasets/cv-corpus-7.0-2021-07-21/ -o /media/dominik/Project-M1nk/datasets-eval/cv-corpus-7.0-2021-07-21/ -ar 16000 -t 24 --lang en
```

## SLR 51 (TED-LIUM Release 3) Dataset
Available here: http://www.openslr.org/51/

Used Version: 3; November 2021

Applicable versions: 3

### What I did
I adapted the encoder preprocess script to also save the transcripts if the proper flag is set and the limits are set high enough to copy over each speaker, even those with few utterances.

*Create sub-utterances and transcript files for each speaker within the target directory*
```
python scripts/tedlium.py /media/dominik/Project-M1nk/datasets/TEDLIUM_release-3/data/ -o /media/dominik/Project-M1nk/datasets-eval/TEDLIUM_release-3/speakers -ft 1 -t 8 --min 1 --max 9999
```

## VCTK-Corpus
Available here: https://datashare.ed.ac.uk/handle/10283/3443

Used Version: 0.92; November 2021

Applicable versions: N/A

### What I did
I adapted the encoder preprocess script to also save the transcripts if the proper flag is set and the limits are set high enough to copy over each speaker, even those with few utterances. It requires the transcript files to be merged into the audio folder, which can be achieved via a simple bash script.

`cp -R txt/* wav48_silence_trimmed/`

*Copy over all utterance and transcript files for each speaker to the target directory*
```
python scripts/vctk.py /media/dominik/Project-M1nk/datasets/VCTK-Corpus/wav48_silence_trimmed/ -o /media/dominik/Project-M1nk/datasets-eval/VCTK-Corpus/speakers/ -ft 1 -t 8 --min 1 --max 9999
```


## Copyright notes

This readme is brought to you by (c) 2021 @RuntimeRacer

Scripts described may be originally forked from @sberryman's WIP branch: https://github.com/sberryman/Real-Time-Voice-Cloning/tree/wip