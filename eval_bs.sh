#!/bin/bash

# Load OpenAI keys.
export OPENAI_API_KEY=$
export LMUData='/export/share/jieyu/LMUData' ## directory to save all datasets
export HF_HOME=/export/share/jieyu/cache/

# Run evaluation on a single GPU.
## Command line args:
## data: A list of evaluation benchmarks, feel free to change the list.
## model: The name of the model to be evaluated. Currently the path to local model checkpoint needs to be specified in ./vlmeval/config.py.
## nproc: If set larger than 1, might get stuck at OpenAI API calls. (Occurs randomly, not every time.)



cd /export/share/jieyu/VLMEvalKit-mantis
source /export/share/zixianma/miniconda/bin/activate
conda activate mantis-eval

for m in v3-both-0.1 v3-mc-0.1 v3-0.1
do
  python run.py --data MMT-Bench_VAL_MI BLINK RealWorldQA MME LLaVABench MMVet SEEDBench_IMG MME MMBench_DEV_EN POPE HallusionBench ScienceQA_TEST ChartQA_TEST DocVQA_VAL MathVista_MINI TextVQA_VAL MMMU_DEV_VAL\
      --model /export/share/jieyu/mantis_ckpt/Mantis-8B-siglip-llama3-pretraind/$m/checkpoint-final --verbose --nproc 4;
done

python3 compile.py --resdir /export/share/jieyu/mantis_ckpt/Mantis-8B-siglip-llama3-pretraind
