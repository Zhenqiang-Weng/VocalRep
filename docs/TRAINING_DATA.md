# Preparing Training Data

This document describes the actual data-loading behavior implemented by
`utils/dataset.py` and `utils/dataset_with_spk.py` in this repository. The meanings of
the `dataset_type` values may differ from those used by other MSS projects.

## 1. Check the configuration first

Stem filenames are determined by `training.instruments` in the selected configuration.
For example, the current `ckpts/multi_stem/config.yaml` contains:

```yaml
audio:
  sample_rate: 44100
  chunk_size: 529200

training:
  instruments: [vocals, backing_vocal, instrumental]
```

Every track must therefore use exactly these case-sensitive stem names. `chunk_size` is
measured in samples, not seconds; calculate its duration as
`chunk_size / sample_rate`. The example above produces 12-second chunks.

## 2. Choose a dataset type

| Type | On-disk layout | Loading behavior | Intended use |
| --- | --- | --- | --- |
| `1` | `root/<track>/<stem>.wav` or `.flac` | Selects a track and offset independently for each stem | Random, unaligned mixing |
| `2` | `root/<stem>/*.wav` or `.flac` | Samples each stem independently from its own pool | Unpaired stem pools |
| `3` | One CSV with `instrum,path` columns | Samples each stem independently from the CSV | Files stored in unrelated locations |
| `4` | `root/<track>/<stem>.wav` or `.flac` | Loads aligned chunks from the same track and offset | Aligned training and speaker conditioning |

The current speaker-conditioned training scripts use `MSSDatasetWithSpk`. Only
`dataset_type=4` loads real values from `embeddings.npy`; types 1–3 return an all-zero
192-dimensional embedding. Type 4 is therefore recommended when training `spk_*`
models.

The two dataset classes handle `training.mix_instruments` differently. The regular
`MSSDataset` attempts to load stems listed in `mix_instruments` but not in
`training.instruments` as additional mixture-only sources; they are not training
targets and must follow the selected dataset layout. With type 2, `mix_instruments`
must also include every target instrument, or the generated metadata will be
incomplete. `MSSDatasetWithSpk` explicitly ignores this extension, so its mixture is
the sum of `training.instruments` only. The two `train_accelerate_bf16*.py` entry points
currently use `MSSDatasetWithSpk`.

## 3. Recommended type 4 layout

```text
/absolute/path/to/train/
├── song_001/
│   ├── vocals.wav
│   ├── backing_vocal.wav
│   ├── instrumental.wav
│   └── embeddings.npy
├── song_002/
│   ├── vocals.flac
│   ├── backing_vocal.flac
│   ├── instrumental.flac
│   └── embeddings.npy
└── ...
```

All stems within the same track directory should meet these requirements:

- Use the same sample rate; 44.1 kHz is required by the current configuration. The
  loader does not resample automatically.
- Have the same start point, duration, and frame count so they are aligned at the
  sample level.
- Prefer stereo audio. Mono audio is duplicated to stereo, while audio with more than
  two channels is truncated to its first two channels.
- Contain finite, non-silent audio and no NaN or Inf values.
- Use a lowercase `.wav` or `.flac` extension.

A missing stem can cause a loading failure. Silent material may be sampled repeatedly
by the random dataset types while the loader searches for a usable chunk.

### Speaker embeddings

Each track's `embeddings.npy` must be a numeric, two-dimensional array:

```text
shape = (N, 192), N >= 1
dtype = float32 (recommended)
```

During training, the loader randomly selects up to 20 rows and averages them. During
validation, it averages all rows. The embeddings should be generated from clean vocal
segments belonging to the target speaker for that track. This repository does not
include a verified end-to-end embedding-generation workflow. Use a trusted CAMPPlus
extraction environment and manually check speaker identity on a small sample before
preparing the full dataset.

## 4. Other training layouts

### Type 1: track directories with random, unaligned mixing

The directory layout is the same as type 4, but the loader chooses a track and offset
independently for each stem. Stems are therefore not kept paired during training even
when they are aligned by track on disk. Stems inside each track directory should still
have identical frame counts: metadata records the shortest stem length for every file
in that directory, and mismatched files shorter than one `chunk_size` may produce
chunks with invalid lengths.

### Type 2: independent stem pools

```text
/absolute/path/to/train/
├── vocals/
│   ├── vocal_001.wav
│   └── vocal_002.flac
├── backing_vocal/
│   └── backing_001.wav
└── instrumental/
    └── instrumental_001.wav
```

Each instrument in the configuration must have at least one readable file.

### Type 3: CSV index

The current implementation accepts exactly one CSV at a time. The CSV must contain at
least the `instrum` and `path` columns; their spelling and capitalization must match
exactly. Additional columns are ignored:

```csv
instrum,path
vocals,/absolute/path/to/vocal_001.wav
backing_vocal,/absolute/path/to/backing_001.wav
instrumental,/absolute/path/to/instrumental_001.wav
```

Absolute paths are recommended. Relative paths are resolved against the process's
working directory when training starts, not against the directory containing the CSV.
Every configured instrument must have at least one row.

## 5. Validation-set layout

The validation loader always scans for `<valid_root>/<track>/mixture.wav`. It does not
recognize a FLAC mixture or a deeper directory hierarchy. Every validation track must
also contain a WAV file for each configured stem:

```text
/absolute/path/to/valid/
└── song_101/
    ├── mixture.wav
    ├── vocals.wav
    ├── backing_vocal.wav
    ├── instrumental.wav
    └── embeddings.npy
```

`mixture.wav` should have the same sample rate, frame count, and start point as its
stems, and should be the sum of the input stems defined by the training configuration.
Saving it as 32-bit float WAV is recommended to avoid integer PCM clipping during
summation. If the configuration enables `other_fix` and includes an instrument named
`other`, validation uses `mixture - vocals` as the reference for that instrument.

## 6. Validate before training

After installing the training dependencies, run the repository's validator:

```bash
python scripts/validate_training_data.py \
  --config-path ckpts/multi_stem/config.yaml \
  --dataset-type 4 \
  --data-path /absolute/path/to/train \
  --valid-path /absolute/path/to/valid \
  --require-embeddings
```

The validator checks directory and filenames, audio headers, sample rates, frame-count
alignment within a track, CSV columns, validation mixtures, and the numeric dtype,
`(N, 192)` shape, and finite values of embeddings. For a large dataset, you can first
inspect a sample:

```bash
python scripts/validate_training_data.py \
  --config-path ckpts/multi_stem/config.yaml \
  --dataset-type 4 \
  --data-path /absolute/path/to/train \
  --valid-path /absolute/path/to/valid \
  --require-embeddings \
  --max-tracks 20
```

For type 3, `--max-tracks 20` checks at most 20 rows per instrument. Instrument
coverage is still checked against the complete CSV, so a CSV grouped by instrument
will not incorrectly report later stems as missing.

`train_accelerate.sh` automatically performs a complete validation pass before it
starts training.

## 7. Metadata cache

The training loader automatically creates the following file under `results_path`:

```text
metadata_<dataset_type>.pkl
```

Do not edit this file manually. If you add, replace, move, or remove audio files, stop
training, delete the corresponding metadata cache, and then restart training. Cache
reuse is based on paths and does not reliably detect changes to file lengths or
modification times.

## 8. Configure the training Bash script

Edit these values near the top of `train_accelerate.sh`:

```bash
TRAIN_SCRIPT="train_accelerate_bf16.py"
MODEL_TYPE="spk_bs_roformer_exportable"
CONFIG_PATH="ckpts/multi_stem/config.yaml"
RESULTS_PATH="results/multi_stem"
DATASET_TYPE=4
TRAIN_DATA_PATHS=("/absolute/path/to/train")
VALID_DATA_PATHS=("/absolute/path/to/valid")
START_CHECKPOINT=""  # An empty string starts training from scratch.
GPU_IDS=(0)           # Multi-GPU example: (0 1 2 3)
DETERMINISTIC=false   # true improves reproducibility but may reduce performance.
```

Then run:

```bash
conda activate mss
python -m pip install -r requirements-train.txt
bash train_accelerate.sh
```

For online W&B logging, never place the API key in the script. Set it through an
environment variable instead:

```bash
export WANDB_API_KEY='set-this-securely-on-your-machine; do-not-commit-it'
export WANDB_MODE=online
bash train_accelerate.sh
```

Only use datasets and model weights for which you have the necessary rights and
permissions. Do not commit training data, generated metadata, or new weights directly
to ordinary Git history.
