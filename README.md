# distilling-the-essence
Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation

Code for paper [Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation](https://arxiv.org/abs/2512.21002)

# Usage

## Environment Setup
**Prepare the environment**:
  Run the preparation script to set up virtual environments. Pass your work directory (for models/datasets/venvs) and the codebase directory as arguments:

  ```bash
  bash examples/distillation/prepare_env.sh \
      --work-dir "/path/to/your/work_dir" \
      --codebase-dir "/path/to/distilling-the-essence"
  ```

  Arguments:
  - `--work-dir`: Directory where virtual environments are created, and HuggingFace models/datasets are downloaded/cached.
  - `--codebase-dir`: Directory where this repository is located (used to locate `requirements_*.txt`).

  This script creates three virtual environments:
  - One for downloading HuggingFace models and datasets.
  - One for the training environment.
  - One for the evaluation environment.

  The script is designed for environment running Slurm and Lmod. For example, it relies on `StdEnv/2023` ([Alliance Canada Standard Software Environments](https://docs.alliancecan.ca/wiki/Standard_software_environments)). Users on other systems may need to install compatible system dependencies (e.g. gcc).

## Example usages

To run the distillation pipeline, you can use the provided `pipeline.sh` script. Below is an example of running the pipeline with `lsp` (Lead-Span Proportion) set to 0.5, using the Bespoke dataset.

```bash
bash examples/distillation/pipeline.sh \
    --work-dir "/path/to/your/work_dir" \
    --codebase-dir "/path/to/distilling-the-essence" \
    --lsp 0.5
```

Arguments:
- `--work-dir`: Directory where produced files, virtual environments, and caches are stored.
- `--codebase-dir`: Directory where this codebase is stored.
- `--lsp`: Lead-Span Proportion ([0.0, 1.0]). This controls the fraction of the whole sequence to retain from the beginning. For example, `0.5` means keeping the first 50% of the full sequence.

## Reproducing our experiments

We provide data modules for several datasets used in our experiments, including:
- **OpenThoughts**
- **Bespoke-Stratos**
- **Synthetic-1**
- **Nemotron**
- **SkyT1**

You can leverage these included data modules for your experiments. To switch between datasets or modify experiment parameters, you can adjust the environment variables or arguments in `examples/distillation/training.sh` (e.g., setting `dataset="bespoke_stratos17k"` or `dataset="open_thoughts114k"`).

For example, to reproduce experiments for **Bespoke-Stratos-17k**, run:

```bash
# LSP=1.0 (no truncation)
bash examples/distillation/pipeline.sh \
    --work-dir "/path/to/your/work_dir" \
    --codebase-dir "/path/to/distilling-the-essence" \
    --lsp 1.0
```

```bash
# LSP=0.5 (keep first 50% of tokens)
bash examples/distillation/pipeline.sh \
    --work-dir "/path/to/your/work_dir" \
    --codebase-dir "/path/to/distilling-the-essence" \
    --lsp 0.5
```

## Cite
Please consider citing if our paper/code is helpful to you:
```bibtex
@misc{chen2025distillingessenceefficientreasoning,
      title={Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation},
      author={Wei-Rui Chen and Vignesh Kothapalli and Ata Fatahibaarzi and Hejian Sang and Shao Tang and Qingquan Song and Zhipeng Wang and Muhammad Abdul-Mageed},
      year={2025},
      eprint={2512.21002},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.21002},
}
```
