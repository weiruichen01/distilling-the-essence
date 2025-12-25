#!/usr/bin/env bash
set -euo pipefail

repo_name=prepare_for_distilling_the_essence
scratch_dir=/scratch/$USER/$repo_name # where the venv is installed and where models and datasets are downloaded to
code_dir=/home/$USER/$repo_name
mkdir -p $scratch_dir/hf_models
mkdir -p $scratch_dir/hf_datasets

rebuild_env_hf_download=1
download_models=1
download_datasets=1
rebuild_env_exp=1
rebuild_env_eval=1

module load arrow

# ------------------------------------------------------------ #
# venv setup for hf_download
# ------------------------------------------------------------ #
env_hf_download="$scratch_dir/env_hf_download"
if [ "$rebuild_env_hf_download" -eq 1 ]; then
    deactivate 2>/dev/null || true
    if [ -d "$env_hf_download" ]; then
        rm -rf $env_hf_download
        echo " Removed existing virtual environment at $env_hf_download"
    else
        echo " No existing virtual environment at $env_hf_download"
    fi
    echo "Creating virtual environment at $env_hf_download"
    python -m venv $env_hf_download
    echo " Virtual environment created"

    # Activate virtual environment
    echo "Activating virtual environment"
    source $env_hf_download/bin/activate

    # Upgrade pip and install required packages
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install transformers==4.53.1 datasets torch accelerate huggingface_hub
    echo " Required packages installed"
    echo ""
fi

# ------------------------------------------------------------ #
# download models and datasets
# ------------------------------------------------------------ #
models_to_download=(
    'Qwen/Qwen3-32B'
    'Qwen/Qwen3-8B'
    'Qwen/Qwen3-8B-Base'
    'Qwen/Qwen3-4B-Base'
    'Qwen/Qwen3-1.7B'
    'Qwen/Qwen3-0.6B-Base'
)

datasets_to_download=(
    'bespokelabs/Bespoke-Stratos-17k'
    'open-thoughts/OpenThoughts-114k'
    'HuggingFaceH4/aime_2024'
    'yentinglin/aime_2025'
    'NovaSky-AI/Sky-T1_data_17k'
)

if [ "$download_models" -eq 1 ]; then
    source $env_hf_download/bin/activate
    echo "Downloading models to $scratch_dir/hf_models ..."
    export HF_HOME=$scratch_dir/hf_models
    for model in "${models_to_download[@]}"; do
        hf download $model
    done
fi

if [ "$download_datasets" -eq 1 ]; then
    source $env_hf_download/bin/activate
    echo ""
    echo "Downloading datasets to $scratch_dir/hf_datasets ..."
    export HF_HOME=$scratch_dir/hf_datasets
    # for dataset in "${datasets_to_download[@]}"; do
    #     hf download $dataset --repo-type dataset
    # done
    configs=(
    ''
    )
    for d in "${datasets_to_download[@]}"; do
        for config in "${configs[@]}"; do
            echo "Loading $d with config '$config' using datasets package"
            python -c "import datasets; datasets.load_dataset('$d', '$config')"
        done
    done
fi

# ------------------------------------------------------------ #
# venv setup for experiment
# ------------------------------------------------------------ #
if [ "$rebuild_env_exp" -eq 1 ]; then
    deactivate 2>/dev/null || true
    module load cuda/12.6
    env_exp="$scratch_dir/env_exp"
    if [ -d "$env_exp" ]; then
        rm -rf $env_exp
        echo " Removed existing virtual environment at $env_exp"
    else
        echo " No existing virtual environment at $env_exp"
    fi
    echo "Creating virtual environment at $env_exp"
    python -m venv $env_exp
    echo " Virtual environment created"

    # Activate virtual environment
    echo "Activating virtual environment"
    source $env_exp/bin/activate

    # Upgrade pip and install required packages
    echo "Installing required packages..."
    pip install --upgrade pip

    pip install -r $code_dir/examples/distillation/requirements_training.txt
    cd $code_dir
    pip install -e ".[all]"
    echo " Required packages installed"
    echo ""
fi

# ------------------------------------------------------------ #
# venv setup for evaluation
# ------------------------------------------------------------ #
if [ "$rebuild_env_eval" -eq 1 ]; then
    deactivate 2>/dev/null || true

    # Load modules BEFORE creating venv to avoid interference of path settings
    module purge
    module load StdEnv/2023
    module load python/3.11 scipy-stack gcc arrow cuda/12.9 cudnn opencv rust

    env_eval="$scratch_dir/env_eval"
    if [ -d "$env_eval" ]; then
        rm -rf $env_eval
        echo "Removed existing virtual environment at $env_eval"
    else
        echo "No existing virtual environment at $env_eval"
    fi
    echo "Creating virtual environment at $env_eval"
    python -m venv $env_eval
    echo "Virtual environment created"

    # Activate virtual environment
    echo "Activating virtual environment"
    source $env_eval/bin/activate

    # Upgrade pip and install required packages
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install -r $code_dir/examples/distillation/requirements_eval.txt
    pip install "lighteval[extended-tasks]==0.10.0" # install extended tasks
    pip install more_itertools==10.8.0
    pip install openai==1.99.1

    # May need to copy tinyBenchmarks.pkl from https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/tinyBenchmarks/tinyBenchmarks.pkl
    # to env_eval/lib/python3.11/site-packages/lighteval/tasks/extended/tiny_benchmarks/

fi
