#!/bin/bash

# AI Training Dataset fetch and prepare. (c)2021 RuntimeRacer
#
# This script is intended to fetch all datasets of relevance for this project and prepare them for processing.
# You should specify a target directory on a large storage device. 5+TB recommended.
#
# After downloading, the datasets and files will be automatically organized and set up to match the patterns required by the preprocessing scripts.
#

# License and Copyright information of datasets used:
#
# LibriSpeech: 
# VoxCeleb: https://creativecommons.org/licenses/by/4.0/
# LibriTTS:
# VCTK
# M-AILABS: 
#

# You need to specify a target directory
if [ $# -eq 0 ]; then
    echo "No arguments supplied. Usage: ./fetchDatasets.sh <target-directory>"
    exit
fi

targetDir=$1

voxCelebUser="voxceleb1912"
voxCelebPass="0s42xuw6"

# LibriSpeech - http://www.openslr.org/12/
wget -c -O $targetDir/librispeech-train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c -O $targetDir/librispeech-train-clean-360.tar.gz https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c -O $targetDir/librispeech-train-other-500.tar.gz https://www.openslr.org/resources/12/train-other-500.tar.gz

# VoxCeleb 1 - http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
wget -c -O $targetDir/voxceleb1-partaa https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
wget -c -O $targetDir/voxceleb1-partab https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
wget -c -O $targetDir/voxceleb1-partac https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
wget -c -O $targetDir/voxceleb1-partad https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
wget -c -O $targetDir/voxceleb1_meta.csv https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
if [ ! -f $targetDir/voxceleb1_wav.zip ]; then
    echo "Combining Voxceleb1 files to a ZIP"
    cat $targetDir/voxceleb1-parta* > $targetDir/voxceleb1_wav.zip
fi

# VoxCeleb 2 - http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partaa http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partaa
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partab http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partab
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partac http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partac
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partad http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partad
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partae http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partae
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partaf http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partaf
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partag http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partag
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2-partah http://balthasar.tplinkdns.com/voxceleb/vox1a/vox2_dev_aac_partah
wget --user $voxCelebUser --password $voxCelebPass -c -O $targetDir/voxceleb2_meta.csv https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv
if [ ! -f $targetDir/voxceleb2_aac.zip ]; then
    echo "Combining Voxceleb2 files to a ZIP"
    cat $targetDir/voxceleb2-parta* > $targetDir/voxceleb2_aac.zip
fi

# LibriTTS - http://www.openslr.org/60/
wget -c -O $targetDir/libritts-train-clean-100.tar.gz https://www.openslr.org/resources/60/train-clean-100.tar.gz
wget -c -O $targetDir/libritts-train-clean-360.tar.gz https://www.openslr.org/resources/60/train-clean-360.tar.gz
wget -c -O $targetDir/libritts-train-other-500.tar.gz https://www.openslr.org/resources/60/train-other-500.tar.gz

# VCTK - https://datashare.ed.ac.uk/handle/10283/3443
wget -c -O $targetDir/VCTK-Corpus-0.92.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y

# M-AILABS - https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/
wget -c -O $targetDir/m-ailabs_en_UK.tar.gz https://data.solak.de/data/Training/stt_tts/en_UK.tgz
wget -c -O $targetDir/m-ailabs_en_US.tar.gz https://data.solak.de/data/Training/stt_tts/en_US.tgz

