#!/bin/bash

# Build the Library
mkdir build
cd build || print "unable to create build dir" && exit 1
cmake ../src
make

# => This creates a couple python modules. You need to copy them into your runtime python install dir.
# cp WaveRNNVocoder*.so python_install_directory