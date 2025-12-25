#!/bin/bash

set -ex

###################
# Environment Setup
###################
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
module load StdEnv/2023
module load python/3.10 scipy-stack gcc arrow cuda/12.6 cudnn opencv
source $venv_train_path/bin/activate

######################
# Task Parameters
######################

# Student model
: ${model_name:="Qwen3-4B-Base"}
# replace the value (e.g. "Qwen/Qwen3-8B") with the path where model is stored if running in offline mode
# ex: .../hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
declare -A model_path_dict=(
    ["Qwen3-8B"]="Qwen/Qwen3-8B"
    ["Qwen3-1.7B"]="Qwen/Qwen3-1.7B"
    ["Qwen3-4B-Base"]="Qwen/Qwen3-4B-Base"
    ["Qwen3-0.6B-Base"]="Qwen/Qwen3-0.6B-Base"
)
model_path=${model_path_dict[$model_name]}

: ${dataset:="bespoke_stratos17k"}
# replace the value with the path where dataset is stored if running in offline mode
# ex: .../hub/datasets--bespokelabs--Bespoke-Stratos-17k/snapshots/9e9adba943911a9fc44dffcb30aaa18dc96ae6df/data
declare -A data_path_dict=(
    ["bespoke_stratos17k"]="bespokelabs/Bespoke-Stratos-17k"
)
data_path=${data_path_dict[$dataset]}

# Teacher model
: ${teacher_model_name:="Qwen3-8B"}
teacher_model_path=${model_path_dict[$teacher_model_name]}

optimizer="adamw"
lr=4e-6
keep_sparse=False
enable_distill=True
distill_forward_ratio=-1.0
distill_loss="forward_kl"
sample_method="supervised"
max_new_tokens=4096
return_prompt_input_ids=False
use_lora=False
verify_lora_saving_correctness=False
use_liger=False

max_length=4096
batch_size=1
num_epoch=2
weight_decay=0.05
warmup_ratio=0.1


val_check_interval=0.1
if [ "$dataset" == "open_thoughts114k" ]; then
    n_train=40000
    n_val=1000
elif [ "$dataset" == "bespoke_stratos17k" ]; then
    n_train=8000
    n_val=1000
elif [ "$dataset" == "skyt1" ]; then
    n_train=8000
    n_val=1000
fi


distillation_loss_ratio=0.5
is_reasoning_llm=True
cot_start_token="<think>"
cot_end_token="</think>"
included_first_x_percent=0.5
save_on_best_validation=True
section_inclusion="promptcotAns"

enable_gradient_checkpointing=True
enable_torch_compile=False
sample_temperature=0.8
cpu_offload=False
response_template="</think>"
sample_fraction=1
lora_alpha_to_rank_ratio=2
random_mask_ratio=0
gradient_accumulation_steps=2 # to match the effective batch size of 8, as I use 4 H100
enable_fsdp2=False
tvd_log_scale=False
enable_fp8=False
js_beta=0.5
lora_rank=16
include_prompt_loss=False
compile_distill_loss=False
val_include_distill_loss=False
skl_alpha=0.1
inverse_perplexity=False
truncate_after_think_end_token=False

######################
# Run training script
######################

python -u $PROJ_CODE_DIR/examples/distillation/training.py \
    --model_path "$model_path" \
    --teacher_model_path "$teacher_model_path" \
    --dataset "$dataset" \
    --data_path "$data_path" \
    --max_length "$max_length" \
    --batch_size "$batch_size" \
    --output_dir "$OUTPUT_DIR" \
    --optimizer "$optimizer" \
    --lr "$lr" \
    --num_epoch "$num_epoch" \
    --weight_decay "$weight_decay" \
    --warmup_ratio "$warmup_ratio" \
    --n_train "$n_train" \
    --n_val "$n_val" \
    --val_check_interval "$val_check_interval" \
    --enable_distill "$enable_distill" \
    --forward_ratio "$distill_forward_ratio" \
    --distillation_loss_ratio "$distillation_loss_ratio" \
    --distill_loss "$distill_loss" \
    --sample_method "$sample_method" \
    --max_new_tokens "$max_new_tokens" \
    --return_prompt_input_ids "$return_prompt_input_ids" \
    --keep_sparse "$keep_sparse" \
    --use_lora "$use_lora" \
    --use_liger "$use_liger" \
    --verify_lora_saving_correctness "$verify_lora_saving_correctness" \
    --enable_gradient_checkpointing "$enable_gradient_checkpointing" \
    --enable_torch_compile "$enable_torch_compile" \
    --sample_temperature "$sample_temperature" \
    --cpu_offload "$cpu_offload" \
    --response_template "$response_template" \
    --sample_fraction "$sample_fraction" \
    --random_mask_ratio "$random_mask_ratio" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --enable_fsdp2 "$enable_fsdp2" \
    --tvd_log_scale "$tvd_log_scale" \
    --enable_fp8 "$enable_fp8" \
    --js_beta "$js_beta" \
    --lora_alpha_to_rank_ratio "$lora_alpha_to_rank_ratio" \
    --lora_rank "$lora_rank" \
    --include_prompt_loss "$include_prompt_loss" \
    --compile_distill_loss "$compile_distill_loss" \
    --val_include_distill_loss "$val_include_distill_loss" \
    --skl_alpha "$skl_alpha" \
    --inverse_perplexity "$inverse_perplexity" \
    --truncate_after_think_end_token "$truncate_after_think_end_token" \
    --section_inclusion "$section_inclusion" \
    --included_first_x_percent "$included_first_x_percent" \
    --save_on_best_validation "$save_on_best_validation" \
    --is_reasoning_llm "$is_reasoning_llm" \
    --cot_start_token "$cot_start_token" \
    --cot_end_token "$cot_end_token"