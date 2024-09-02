#!/bin/bash

# Load OpenAI keys.
export OPENAI_API_KEY=$2
export LMUData='/export/share/ayan/LMUData' ## directory to save all datasets
export HF_HOME=/export/share/jieyu/cache/
export RUN_NAME=$1

# Run evaluation on a single GPU.
## Command line args:
## data: A list of evaluation benchmarks, feel free to change the list.
## model: The name of the model to be evaluated. Currently the path to local model checkpoint needs to be specified in ./vlmeval/config.py.
## nproc: If set larger than 1, might get stuck at OpenAI API calls. (Occurs randomly, not every time.)



cd /export/share/jieyu/VLMEvalKit-mantis
source /export/share/zixianma/miniconda/bin/activate
conda activate mantis-eval

python run.py --data MMT-Bench_VAL_MI DUDE BLINK \
    --model /export/share/jieyu/mantis_ckpt/Mantis-8B-siglip-llama3-pretraind/$1/checkpoint-final --verbose --nproc 4;

# python3 compile.py --resdir /export/share/ayan/llava_checkpoints/$1
