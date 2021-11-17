#!/bin/bash

python encoder_preprocess.py /media/dominik/Project-M1nk/datasets-ready/ -o ~/linux-workspace/datasets/SV2TTS/encoder -d slr_all:wav,commonvoice_all:flac,nasjonalbank:wav,vctk:flac,voxceleb1:wav,voxceleb2:wav,libritts_other:wav -s -t 12