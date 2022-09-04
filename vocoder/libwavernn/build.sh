#!/bin/bash

# Get required packages
sudo apt-get install cmake gcc libeigen3-dev python3-dev python3-distutils

# Clean dirs & Build the Library
rm -rf build
mkdir -p build
cd build || (print "unable to create build dir" && exit 1)

# If using standalone Python env
#cmake ../src
# If using Conda
# https://cmake.org/cmake/help/git-stage/module/FindPython3.html
# https://conda-forge.org/docs/maintainer/knowledge_base.html
cmake ../src -DPYTHON_EXECUTABLE=/home/dominik/anaconda3/envs/kajispeech-realtime/bin/python

make

# => This creates a couple python modules. You need to copy them into your runtime python install dir.
# cp WaveRNNVocoder*.so python_install_directory