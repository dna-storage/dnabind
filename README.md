# 🧬 Reading between the strands: biophysically informed sequence encodings for weak-affinity DNA–DNA binding prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](https://pytorch.org/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


> 📄 This is the official code release for the paper **“Reading between the
> strands: biophysically informed sequence encodings for weak-affinity DNA–DNA
> binding prediction”**. The 12 encoders benchmarked in
> the paper are exactly the ones shipped here — see [Encoders](#encoders-).


## Table of Contents

- [Overview](#overview-)
- [Installation](#installation-)
  - [Prerequisites](#prerequisites)
  - [One command (recommended)](#one-command-recommended)
  - [Manual install](#manual-install-equivalent-step-by-step)
- [Quickstart](#quickstart-)
- [Encoders](#encoders-)
- [Architectures](#architectures-)
- [Data format](#data-format-)
- [Synthetic datasets](#synthetic-datasets-)
  - [Cross-validation splits](#cross-validation-splits)
- [Predict on a single pair](#predict-on-a-single-pair-)
- [Extending dnabind](#extending-dnabind-)
- [Training outputs](#training-outputs-)
- [Notes](#notes-)
- [Citation](#citation-)

## Overview ✨

Predicting whether two DNA molecules bind underlies core methods across
molecular biology, diagnostics, high-throughput bioengineering, and emerging
technologies such as DNA-based data storage. The hard part is **weak-affinity**
binding — partially complementary strands that deviate from perfect Watson–Crick
pairing — where sequence-distance and thermodynamic heuristics fall short and
deep learning now leads.

The accompanying paper asks how much the **input encoding** matters here. Rather
than leaving a one-hot encoder to learn hybridization rules entirely from data,
it proposes **biophysically informed encoders** that embed Watson–Crick and
nearest-neighbor thermodynamic priors directly into the input tensor, and
benchmarks 12 distinct encoders. The biophysically
informed encoders generalize better and are more data-efficient and robust — up
to **0.22 higher AUROC** and **>10× data efficiency** under limited training
data — at a modest 3–5× resource overhead that still fits on a commodity GPU
(under 4 GB peak memory).

This repository is the official implementation accompanying the paper — for simplicity we'll refer to it as `dnabind` (also the name of its CLI and Python package). It lets you train and evaluate CNN models for DNA–DNA binding prediction, with
**pluggable sequence-pair encoders**.

A binding example is a pair of DNA sequences `(Seq1, Seq2)` with a binary
`Label` (`1` = bound, `0` = unbound). This repository ships with a family of built-in
encoders and two model families, and is designed so that **adding a new encoder
is a single self-contained file**.

This project is built around two swappable pieces:

- **🧩 Encoders** turn a `(seq1, seq2)` pair into a fixed-shape tensor. Each is a
  single self-contained file; the output shape decides which model it feeds.
- **🧱 Architectures** are described entirely in JSON — no code to edit the layer
  stack. Two CNN families ship in the box, one for each shape of encoder output:
  - **`cnn_4xn`** — for the one-hot-style `(C, 4, W)` inputs. A 2D convolution
    first collapses the height-4 (A/C/G/T) axis, then a 1D convolution stack runs
    along the sequence-length axis, followed by a linear head.
  - **`cnn_nxn`** — for the square `(1, L, L)` pair-matrix inputs. A 2D
    convolution stack scans the `L×L` grid directly, followed by a linear head.


Everything is driven through one CLI (`dnabind train | eval | list-encoders |
list-models`).

## Installation 🛠️

You can expect setup to take about **5–10 minutes**, mostly PyTorch downloading.

### Prerequisites

Before you begin, make sure you have:

- **Git** — to clone the repository.
- **Conda** (Miniconda/Anaconda) — Python environment manager.
  [Get Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- **Make** — A build automation tool, typically pre-installed on Linux and macOS. For Windows, you might need to install it (e.g., via Chocolatey or Git Bash which often includes it).
- **OS Compatibility**: This package has been tested and verified to work on the following operating systems:

    | Operating System  | Version(s)       | Architecture           |
    | ----------------- | ---------------- | ---------------------- |
    | Red Hat Enterprise Linux | 9.4        | x86_64                 |
    | macOS (Sequoia)   | 15.6             | arm64 (Apple Silicon)  |

### Step-by-Step Setup Guide 👇

Follow these steps in your terminal or command prompt:

### 1. Clone the Repository

First, get a copy of this project onto your computer:

```bash
git clone https://github.com/dna-storage/dnabind.git
cd dnabind
```
### 2. Initialize the Project Environment (First Time Only!)
#### One command (recommended)

From the repository root:

```bash
make init          # creates the conda env, installs PyTorch, installs dnabind
```

`make init` runs `init.sh`, which (1) creates/updates the `dnabind` conda
environment from the platform's environment file, (2) installs the tested
PyTorch build, and (3) installs the package in editable mode
(`pip install -e .`). It's safe to re-run.

The platform is **auto-detected** from your OS: Linux uses
`environment.linux.yml` + the CUDA 12.6 PyTorch build, and macOS uses
`environment.macos.yml` + the CPU/MPS build (there is no CUDA on Mac).

- **What to expect:** lots of download/install messages; this can take several
  minutes depending on your connection.
- ⚙️ **Override the platform:** if auto-detection is wrong (e.g. a CPU-only
  Linux box), pass it explicitly: `bash init.sh linux` (or
  `DNABIND_PLATFORM=linux bash init.sh`).
- 💡 **When to re-run:** you generally don't need to. Re-run only if the
  environment file changes; for pure source edits the editable install picks
  them up automatically.

#### Manual install (equivalent, step by step)

PyTorch is installed on its own because the right build depends on your
hardware, so it is deliberately **not** pinned in the environment file.

```bash
# 1. Create + activate the conda env (python + numpy/pandas/scikit-learn/matplotlib/jupyter)
#    Pick the file for your OS: environment.linux.yml or environment.macos.yml
conda env create -f environment.linux.yml   # macOS: environment.macos.yml
conda activate dnabind

# 2. Install PyTorch — pick the line for your machine. Don't execute both lines
# CPU only / macOS
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0   
# CUDA 12.6
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126   

# 3. Install dnabind itself (torch already satisfied, so no second copy is pulled)
pip install -e .
```

### 3. Activate Your New Environment (Every Session!)

After `make init` finishes, the environment is created, but it's not automatically active in your current terminal session. You need to activate it manually.

**You must run this command in your terminal every time you open a new terminal session to work on this project:**

```bash
conda activate dnabind
```

- **`conda activate dnabind`**: This command switches your terminal to use the Python and packages from the `dnabind` Conda environment. You'll usually see `(dnabind)` appear at the beginning of your terminal prompt when it's active.

## Quickstart 🚀

Runs end to end on the bundled  demo data.

```bash
# 2. Train + evaluate a 4×N CNN on the interleaved one-hot encoder
dnabind train \
  --data_dir data/demo \
  --encoder onehot_interleaved_reversed \
  --arch_config configs/cnn_4xn_small.json \
  --checkpoint experiments/demo/best_model.pt \
  --log_dir experiments/demo \
  --epochs 2 --eval

# ...or an N×N CNN on the nearest-neighbor ΔG pair-matrix encoder
dnabind train \
  --data_dir data/demo \
  --encoder nearestneighbor_mismatch_reversed \
  --arch_config configs/cnn_nxn_small.json \
  --checkpoint experiments/demo_nxn/best_model.pt \
  --log_dir experiments/demo_nxn \
  --epochs 2 --eval
```

Evaluate a saved checkpoint later (the encoder is remembered by the checkpoint):

```bash
dnabind eval --data_dir data/demo --checkpoint experiments/demo/best_model.pt \
  --log_dir experiments/demo
```

List what's available:

```bash
dnabind list-encoders     # all names in the Encoders table below
dnabind list-models       # model families: cnn_4xn, cnn_nxn
```

## Encoders 🧩

Every encoder turns a `(seq1, seq2)` pair into a fixed-shape tensor. The output
shape determines which model family it pairs with: a `(C, 4, W)` shape feeds
`cnn_4xn`, a square `(1, L, L)` shape feeds `cnn_nxn`.

These are the **12 encoders evaluated in the paper**: three one-hot baselines
(interleaved, concatenated, dual-channel) and the biophysically informed
pair-matrix encoders (binary and weighted Watson–Crick, and nearest-neighbor
ΔG), most available in both `native` and `_reversed` forms.

**Reversed convention.** Most encoders come in two registered names: the `native`
name uses `seq2` as given (5'→3'), and the `_reversed` name reverses `seq2`
first so that a perfect antiparallel duplex lines up. The
nearest-neighbor encoders are only physically meaningful in the reversed
forms, so only their `_reversed` form is provided.

| Encoder name | Output shape | Model family | Description |
|---|---|---|---|
| `onehot_interleaved` / `onehot_interleaved_reversed` | `(1, 4, 2L)` | `cnn_4xn` | One-hot; seq1/seq2 bases interleaved column by column |
| `onehot_concatenated` / `onehot_concatenated_reversed` | `(1, 4, 2L)` | `cnn_4xn` | One-hot; seq1 block followed by seq2 block |
| `onehot_dual_channel` / `onehot_dual_channel_reversed` | `(2, 4, L)` | `cnn_4xn` | One-hot; seq1 and seq2 as two separate channels |
| `wc_binary` / `wc_binary_reversed` | `(1, L, L)` | `cnn_nxn` | 1.0 at Watson–Crick pairs, else 0 |
| `wc_weighted` / `wc_weighted_reversed` | `(1, L, L)` | `cnn_nxn` | 1.0 at G–C/C–G, 0.5 at A–T/T–A, else 0 |
| `nearestneighbor_reversed` | `(1, L, L)` | `cnn_nxn` | Nearest-neighbor ΔG (kcal/mol) at perfect NN steps |
| `nearestneighbor_mismatch_reversed` | `(1, L, L)` | `cnn_nxn` | Nearest-neighbor ΔG scoring perfect, single- and double-mismatch steps |

`L` is the sequence length (`--seq_length`, default 20).

> 💡 **See the encodings for yourself.** `notebooks/visualize_encodings.ipynb`
> runs every encoder on one pair, prints the arrays, and draws an annotated
> heatmap gallery inline — handy for building intuition. Edit `seq1`/`seq2` at the
> top and re-run.

## Architectures 🧱

Architectures are plain JSON. The `model_type` field selects the family; the
rest describes the layers. See `configs/` for complete, runnable examples.

```json
{
  "model_type": "cnn_nxn",
  "lr": 0.001,
  "batch_size": 512,
  "conv2d_layers_config": [
    {"out_channels": 64, "kernel_size": 3, "padding": 1,
     "activation": "relu", "batch_norm": true, "dropout": 0.1}
  ],
  "linear_layers_config": [{"out_features": 128, "activation": "relu"}, {"out_features": 1}],
  "final_global_pool": null
}
```

- **`lr` and `batch_size` are required keys** and are read from the JSON — keeping every training run's hyperparameters in one
  reproducible file.
- Input dimensions are injected from the encoder's `output_shape`, so
  architecture JSONs never hard-code tensor sizes.

## Data format 📁

CSV files named `train.csv`, `val.csv`, `test.csv` in one directory, each with
these columns:

| Seq1 | Seq2 | Label |
|------|------|-------|
| 20-mer (5'→3') | 20-mer (5'→3') | `0` or `1` |

Sequences are fixed-length (default 20; set with `--seq_length`). The current implementation only accepts 20bp long sequences.

## Synthetic datasets 🧪

Beyond your own CSVs, `dnabind.datagen` ships **two dataset generators** that
build synthetic DNA-pair datasets in the `Seq1, Seq2,
Label` format above (plus extra descriptive columns):

| Generator | Module | What it builds |
|---|---|---|
| `complementary` | `dnabind.datagen.complementary` | Random **perfect-complementary** pairs — `seq2 = RC(seq1)`, a blunt 20-bp Watson–Crick duplex|
| `offset` | `dnabind.datagen.offset` | An **offset ladder** of staggered antiparallel duplexes; each backbone appears at every signed offset for paired breaking-point analysis |

Each module runs standalone and writes a CSV:

```bash
python -m dnabind.datagen.complementary --n-pairs 500 --seed 0 \
  --out data/complementary/complementary.csv

python -m dnabind.datagen.offset --n-backbones 20 --min-offset 5 --max-offset 10 \
  --seed 0 --out data/offset_ladder/offset_ladder.csv
```

> 💡 **Worked example.** `notebooks/dataset_generation.ipynb` walks through an example
> generator end to end — calling, inspecting the constructs, and saving the
> resulting datasets.

### Cross-validation splits

For the generalization studies, two helper scripts in `scripts/`
carve master `train/val/test.csv` files (each row tagged with an `exp_index` in
`2A..2Z`) into ready-to-train dataset directories. They stream the CSVs in chunks
so datasets far larger than memory are fine, and self-check the outputs (schema,
row counts, no train/test leakage).

| Script | Builds | For |
|---|---|---|
| `scripts/generate_loo_splits.py` | 26 **leave-one-experiment-out** dirs: train/val exclude experiment `X`, test is all of `X` | Train on 25 experiments, test on the held-out 1 |
| `scripts/generate_train_one_splits.py` | 26 **train-on-one** dirs: a train/val split of a single experiment `X` | Train on 1 experiment, test on the other 25 |

```bash
# 26 leave-one-experiment-out directories (2A..2Z) from master splits
python scripts/generate_loo_splits.py \
  --master_dir data/binding_dataset \
  --out_dir    data/binding_dataset_loo

# 26 train-on-one directories, reusing each LOO dir's single-experiment test.csv
python scripts/generate_train_one_splits.py \
  --loo_dir data/binding_dataset_loo \
  --out_dir data/binding_dataset_trainone \
  --val_frac 0.1
```

## Predict on a single pair 🔮

Checkpoints are self-describing — they remember which encoder and sequence
length they were trained with, so loading is a one-liner:

```python
from dnabind import load_model

model = load_model("experiments/demo/best_model.pt")
label, prob = model.predict("AGCGATACGCCTTAACGTCT", "AATGGCGAAGGGGATCGTTC")
print(label, prob)   # e.g. "Unbound" 0.03
```

See `notebooks/playground.ipynb` for an interactive version.

## Extending dnabind 🧰

The framework is modular by design — add a new encoding scheme or a whole new
model family without touching the training loop.

### Add a new encoder

Drop one file in `src/dnabind/encoders/`. It's picked up automatically — no
registration list to edit:

```python
# src/dnabind/encoders/my_encoder.py
import numpy as np
from .base import Encoder, register_encoder

@register_encoder("my_encoder")
class MyEncoder(Encoder):
    def output_shape(self, seq_length):
        return (1, seq_length, seq_length)      # (C, H, W)
    def encode(self, seq1, seq2, seq_length):
        ...                                      # -> np.ndarray of that shape
```

An `(C, 4, W)` encoder pairs with the `cnn_4xn` family; a square `(C, L, L)`
encoder pairs with `cnn_nxn`.

### Add a new architecture family

Register an `nn.Module` with a `from_config(config, input_shape)` classmethod
via `@register_model("my_family")` in `src/dnabind/models/`, then select it from
any architecture JSON with `"model_type": "my_family"`.

## Training outputs 📂

Each run writes to its `--log_dir`:

| File | Contents |
|------|----------|
| `epoch_summary.csv` | Per-epoch training/validation metrics, including per-epoch wall-clock time (`epoch_time_s`) |
| `training_stats.json` | Whole-run resource stats: total training time (`total_training_time_s`) and, on CUDA, peak GPU memory (`peak_gpu_memory_bytes` / `peak_gpu_memory_mib`) |
| `test_log.csv` | Per-sample predictions (with `--eval`) |
| `test_metrics.json` | Accuracy / precision / recall / F1 / AUC (with `--eval`) |
| `roc_curve.png` | ROC curve (with `--eval`) |

The best checkpoint is saved separately to `--checkpoint`.

## Notes 📝

- **Local by default.** Training uses CUDA if available, else CPU. The CLI is
  fully flag-driven, so it can be wrapped by an HPC scheduler without code
  changes.

## Citation
> ℹ️ Full citation details (author list, journal, year, DOI) will be finalized
> on publication.
