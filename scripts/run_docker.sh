#!/bin/bash

# Run docker image with repo attached
docker run --gpus all --ipc=host -p 8888:8888 --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v "$PWD":/workspace bscore/pytorch:bscore
