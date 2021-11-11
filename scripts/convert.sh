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

total=$(find . -iname "*.m4a" -o -iname "*.mp3" -o -iname "*.wav" | wc -l)
skipped=0
converted=0

N=12 # number of vCPU
open_sem $N
for f in $(find . -iname "*.m4a" -o -iname "*.mp3" -o -iname "*.wav"); do 
    if [ -f "${f%.*}.flac" ]; then
        let skipped=skipped+1
        continue
    fi
    run_with_lock ffmpeg -loglevel panic -i "$f" -c:a flac -compression_level 12 -ar 24000 "${f%.*}.flac"
    let converted=converted+1
    echo -ne "Total files: $total. Skipped $skipped already converted files; converted $converted new files."\\r
done

echo "Done converting audio files. Cleaning up..."

find . -iname "*.m4a" -o -iname "*.mp3" -o -iname "*.wav" -delete
echo Deleted $total files which were converted successfully.



# deleted=0

# for f in $(find . -name "*.m4a"); do 
#     if [ -f "${f%.*}.wav" ]; then
#         run_with_lock rm "$f"
#         let deleted=deleted+1
#     fi
#     echo -ne "Total files: $total. Deleted $deleted files which were converted successfully."\\r
# done