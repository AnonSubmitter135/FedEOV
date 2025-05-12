# Anonymous Submission - Federated Ensemble with OSR

This repository contains the codebase for our NeurIPS 2025 anonymous submission on federated learning with ensemble-based open-set recognition. It implements a variety of federated learning algorithms and provides a unified training and evaluation workflow.

The core training pipeline is defined in `main.py`, which orchestrates data partitioning, client/server interactions, model aggregation, and evaluation. All experiments can be configured and launched using the provided `run_code.bat` script.

## Running Experiments

To run experiments, configure your desired settings in the `run_code.bat` file. Then, from the terminal, execute:

```bash
run_code.bat
```

The script iterates over combinations of parameter values and executes each training configuration automatically.

## Parameter Overview

The following parameters are used in `run_code.bat` and passed to `main.py`:

| Parameter             | Description                                | Example Values                                                                 |
|-----------------------|--------------------------------------------|--------------------------------------------------------------------------------|
| `--init_seed`         | Random seed                                | `0`, `1`, `2`                                                                  |
| `--dataset`           | Dataset to use                             | `mnist`, `fashion-mnist`, `colored-mnist`, `svhn`, `cifar10`, `cifar100`      |
| `--batch-size`        | Batch size for training                    | `64`, `128`                                                                    |
| `--cd_lr`             | Learning rate for classifier               | `0.001`, `0.01`                                                                |
| `--partition`         | Data partitioning strategy                 | `homo`, `noniid-labeldir`, `XtremeHetero`                                       |
| `--beta`              | Dirichlet parameter for heterogeneity      | `0.1`                                                                          |
| `--n_parties`         | Number of federated clients                | `10`                                                                           |
| `--algorithm`         | Federated learning algorithm               | `fedavg`, `fedprox`, `scaffold`, `fedsam`, `feddyn`, `fedmoon`, `fedconcat`, `fedov`, `fedeov`, `fedprunedeov`, `distilled_ov`, `distilled_eov`, `centralized` |
| `--rounds`            | Number of federated learning rounds        | `100`                                                                          |
| `--epochs_classifier` | Epochs per round                           | `5`, `10`                                                                      |
| `--model`             | Model architecture                         | `SimpleCNN2`                                                                   |
| `--device`            | Computation device                         | `cpu`, `cuda:0`                                                                |
| Flags                 | Optional toggles                           | `--save_result_dict`, `--save_classifier`, `--ablation`                        |

## Output

- Results are saved in `results/` or `ablation_results/` as `.json` files

## Notes

- This repository has been anonymized for peer review.
- All results reported in the paper are averaged over multiple random seeds.
- All datasets will be automatically downloaded to the directory specified by `--datadir` (default: `./data/`) if not already present.
- Please refer to Appendix B in the paper for exact configurations used in each experimental setting.

