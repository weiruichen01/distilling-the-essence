#!/usr/bin/env bash
set -ex

echo 'entering pipeline.sh'

datetime=`date +%Y%m%d-%H%M%S`
echo "datetime: $datetime"

repo_name=distilling-the-essence

# Replace these variables for your need
PROJ_SCRATCH_DIR="/scratch/$USER/$repo_name" # where produced files are stored
PROJ_CODE_DIR="/home/$USER/$repo_name" # where codebase is stored
account= # slurm account
export venv_train_path=$PROJ_SCRATCH_DIR/env_exp # the path to venv for training
export venv_eval_path=$PROJ_SCRATCH_DIR/env_eval
export HF_CACHE=$PROJ_SCRATCH_DIR/hf_datasets/datasets
export EVAL_OUTPUT_DIR=$PROJ_SCRATCH_DIR/eval_results
export HF_DATASETS_CACHE="$HF_CACHE"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$HF_CACHE"

export model_name=Qwen3-4B-Base
export teacher_model_name=Qwen3-8B
export dataset=bespoke_stratos17k
export do_train=1
export do_eval=1

exp_id=${model_name}_${teacher_model_name}_${dataset}_${datetime}
export OUTPUT_DIR="${PROJ_SCRATCH_DIR}/exp_results/${exp_id}"
mkdir -p "$OUTPUT_DIR"

train_time="25:00:00"
eval_time="23:00:00"

# Note that it has to be '--gpus-per-node=h100:2' not --gpus=h100:2 because
# FSDP requires two GPUs to be on the same node. --gpus=h100:2 cannot guarantee that.
if [ "$do_train" -eq 1 ]; then
    train_job_id=$(sbatch --job-name="train_${exp_id}" \
           --output="${PROJ_SCRATCH_DIR}/eo_files/${exp_id}_train.o" \
           --error="${PROJ_SCRATCH_DIR}/eo_files/${exp_id}_train.e" \
           --gpus-per-node=h100:4 \
           --time=${train_time} \
           --export=ALL \
           --account=${account} \
           --cpus-per-task=12 \
           --mem=1000G \
           --wait \
           --parsable \
           $PROJ_CODE_DIR/examples/distillation/training.sh)
fi

export eval_model_path=$OUTPUT_DIR/final_model

export benchmark="lighteval|aime25|0|0"
if [ "$do_eval" -eq 1 ]; then
    if [ "$do_train" -eq 1 ]; then
        dependency_flag="--dependency=afterok:${train_job_id}"
    else
        dependency_flag=""
    fi
    sbatch --job-name="eval_${exp_id}" \
           --output="${PROJ_SCRATCH_DIR}/eo_files/${exp_id}_eval.o" \
           --error="${PROJ_SCRATCH_DIR}/eo_files/${exp_id}_eval.e" \
           --gpus=h100:1 \
           --time=${eval_time} \
           --export=ALL \
           --account=${account} \
           --cpus-per-task=12 \
           --mem=250G \
           ${dependency_flag} \
           $PROJ_CODE_DIR/examples/distillation/evaluate.sh
fi