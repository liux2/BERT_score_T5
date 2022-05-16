#!/bin/bash

# Run docker image with repo attached
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v "$PWD":/workspace bscore/pytorch:bscore
