#!/bin/bash

# IMPORTANT! - WaveRNN Module variant to build
BUILD_VARIANT="runtimeracer_version"

# Get required packages
sudo apt-get install cmake gcc libeigen3-dev python3-dev python3-distutils zlib1g-dev

# Uninstall any old version of the lib
pip uninstall WaveRNNVocoder

# Clean dirs & Build the Library
rm -rf build
mkdir -p build
cd build || (print "unable to create build dir" && exit 1)

# If using standalone Python env
#cmake ../$BUILD_VARIANT/src
# If using Conda
# https://cmake.org/cmake/help/git-stage/module/FindPython3.html
# https://conda-forge.org/docs/maintainer/knowledge_base.html

# Specific Python version in conda env, will fail if no specific version in your env:
if [[ ! -z $CONDA_PREFIX ]]
then
  cmake ../$BUILD_VARIANT/src -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
else
  if [[ ! -z $CONDA_PYTHON_EXE ]]
  then
    cmake ../$BUILD_VARIANT/src -DPYTHON_EXECUTABLE=$CONDA_PYTHON_EXE
  else # Fallback to system python if conda not detected (https://github.com/RuntimeRacer/Real-Time-Voice-Cloning/issues/9)
    cmake ../$BUILD_VARIANT/src -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
  fi
fi

make

# => This creates a couple python modules in /build.
# Copy the WaveRNNVocoder.so into the 'WaveRNNVocoder' folder and install / uninstall via pip:
cd ..
cp build/WaveRNNVocoder*.so WaveRNNVocoder/WaveRNNVocoder.so
pip install WaveRNNVocoder/.

