#!/bin/bash

echo "Checking NVIDIA CUDA/Drivers"
nvidia-smi
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error using GPU - check if nvidia-driver-package matches with host"
    dpkg -l nvidia-driver* | grep ii
    exit $retVal
fi


source activate_run.sh
export PYTHONPATH=/extract_mkv/build/lib/ 
export CUDA_VISIBLE_DEVICES=0,1
exec python3 /deploy/color_frames.py


while true
do
    echo "Sleeping..."
    sleep 10
done
