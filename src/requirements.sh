#!/bin/bash
echo "This is a script to install all required library for successfully execute all files"
pip install transformers
pip install sentencepiece==0.1.94
pip install tqdm
pip install bert_score
echo "finished!"
