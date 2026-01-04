#!/usr/bin/env bash
set -ex

usage() {
    echo "Usage: $0 --work-dir <path> --codebase-dir <path> --lsp <value>"
    echo ""
    echo "Required arguments:"
    echo "  --work-dir <path>      Directory where produced files, venvs, and caches are stored"
    echo "  --codebase-dir <path>  Directory where codebase is stored"
    echo "  --lsp <value>          Fraction of reasoning tokens to include (0.0-1.0), maps to included_first_x_percent"
    echo ""
    echo "Options:"
    echo "  -h, --help             Show this help message"
    exit 1
}

work_dir=""
codebase_dir=""
lsp=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --work-dir)
            work_dir="$2"
            shift 2
            ;;
        --codebase-dir)
            codebase_dir="$2"
            shift 2
            ;;
        --lsp)
            lsp="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$work_dir" ]]; then
    echo "Error: --work-dir is required"
    usage
fi
if [[ -z "$codebase_dir" ]]; then
    echo "Error: --codebase-dir is required"
    usage
fi
if [[ -z "$lsp" ]]; then
    echo "Error: --lsp is required"
    usage
fi

echo 'entering pipeline.sh'

datetime=`date +%Y%m%d-%H%M%S`
echo "datetime: $datetime"

echo "work_dir: $work_dir"
echo "codebase_dir: $codebase_dir"
echo "lsp: $lsp"

export work_dir
export codebase_dir
export included_first_x_percent=$lsp
account= # slurm account
export venv_train_path=$work_dir/env_exp # the path to venv for training
export venv_eval_path=$work_dir/env_eval
export HF_CACHE=$work_dir/hf_datasets/datasets
export EVAL_OUTPUT_DIR=$work_dir/eval_results
export HF_DATASETS_CACHE="$HF_CACHE"
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HOME="$HF_CACHE"

export model_name=Qwen3-4B-Base
export teacher_model_name=Qwen3-8B
export dataset=bespoke_stratos17k
export do_train=1
export do_eval=1

exp_id=${model_name}_${teacher_model_name}_${dataset}_lsp${lsp}_${datetime}
export OUTPUT_DIR="${work_dir}/exp_results/${exp_id}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "${work_dir}/eo_files"

train_time="25:00:00"
eval_time="25:00:00"

# Note that it has to be '--gpus-per-node=h100:2' not --gpus=h100:2 because
# FSDP requires two GPUs to be on the same node. --gpus=h100:2 cannot guarantee that.
if [ "$do_train" -eq 1 ]; then
    train_job_id=$(sbatch --job-name="train_${exp_id}" \
           --output="${work_dir}/eo_files/${exp_id}_train.o" \
           --error="${work_dir}/eo_files/${exp_id}_train.e" \
           --gpus-per-node=h100:4 \
           --time=${train_time} \
           --export=ALL \
           --account=${account} \
           --cpus-per-task=12 \
           --mem=1000G \
           --wait \
           --parsable \
           $codebase_dir/examples/distillation/training.sh)
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
           --output="${work_dir}/eo_files/${exp_id}_eval.o" \
           --error="${work_dir}/eo_files/${exp_id}_eval.e" \
           --gpus=h100:1 \
           --time=${eval_time} \
           --export=ALL \
           --account=${account} \
           --cpus-per-task=12 \
           --mem=250G \
           ${dependency_flag} \
           $codebase_dir/examples/distillation/evaluate.sh
fi