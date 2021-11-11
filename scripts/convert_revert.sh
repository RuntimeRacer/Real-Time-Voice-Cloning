#!/bin/bash

# Small helper Script to convert files to .wav / .flac format.
# Needs to be put into <path-to-VoxCeleb2>/raw/dev/aac or top level layer of datasets containing folder respectively
#
# TODO: Make this part of bootstrap
#
# For more details, see: https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/488#issuecomment-673944854

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (     
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}            

found=$(find . -iname "*.m4a" -o -iname "*.mp3" -o -iname "*.wav")
total=$($found | wc -l)
skipped=0
deleted=0

N=8 # number of vCPU
open_sem $N
for f in $found; do 
    if [ -f "${f%.*}.flac" ]; then
        let deleted=deleted+1
        run_with_lock rm "${f%.*}.flac"
    else
        let skipped=skipped+1
    fi
    echo -ne "Total files: $total. Skipped $skipped not converted files; deleted $deleted already converted files."\\r
done



# deleted=0

# for f in $(find . -name "*.m4a"); do 
#     if [ -f "${f%.*}.wav" ]; then
#         run_with_lock rm "$f"
#         let deleted=deleted+1
#     fi
#     echo -ne "Total files: $total. Deleted $deleted files which were converted successfully."\\r
# done