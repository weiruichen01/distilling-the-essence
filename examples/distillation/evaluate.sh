#!/usr/bin/env bash
set -euo pipefail

echo -e "\n\nRunning evaluate.sh ...\n\n"

module purge
module load StdEnv/2023
module load python/3.11 scipy-stack gcc arrow cuda/12.9 cudnn opencv rust
source $venv_eval_path/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Running LightEval on ${eval_model_path}"

MODEL_ARGS="model_name=${eval_model_path},dtype=bfloat16,trust_remote_code=true,max_model_length=32768,gpu_memory_utilization=0.8,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

lighteval vllm "$MODEL_ARGS" "$benchmark" \
  --use-chat-template \
  --output-dir "$EVAL_OUTPUT_DIR" \
  --save-details