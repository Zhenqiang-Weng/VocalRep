# Music Source Separation

An experimental PyTorch project for music source separation. The repository
includes BS-Roformer, Mel-Band Roformer, SCNet, and BandIt v2, together with
optional speaker-guided, discriminator, and diffusion components.

> [!IMPORTANT]
> This repository is still being organized as an experimental project. The
> basic inference pipeline and several model implementations are available,
> but the training entry points, speaker-model assets, and some legacy
> discriminator code have not yet been fully validated end to end. Review the
> configuration, dataset paths, and device settings before running them.

## Repository layout

```text
.
├── ckpts/                 # Original configuration files and Git LFS weights
├── models/                # Source-separation models
├── diffusion/             # Optional diffusion components
├── discriminator/         # Optional discriminator components
├── spk_extract/           # CAMPPlus speaker-feature code
├── utils/                 # Data, loss, metric, and inference utilities
├── mss_api/               # Experimental API and export code
├── inference.py           # Basic directory-based inference entry point
├── inference_with_spk.py  # Experimental speaker-guided entry point
└── train_accelerate_*.py  # Accelerate training entry points
```

## Environment setup

Recommended environment:

- Linux or WSL2
- Python 3.10
- An NVIDIA GPU (a CPU is sufficient for basic checks, but inference with
  large models will be slow)
- FFmpeg and Git LFS

Create and activate the environment:

```bash
conda create -n mss python=3.10 -y
conda activate mss
python -m pip install --upgrade pip
```

Install a PyTorch build that matches your hardware first. The command below is
an example for CUDA 13.0. For other platforms, generate the appropriate command
with the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
python -m pip install \
  torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Install the main runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For training, install the additional training dependencies:

```bash
python -m pip install -r requirements-train.txt
```

Install the optional dependencies only if you need the additional optimizers,
legacy discriminator modules, real-time audio support, or ONNX export:

```bash
python -m pip install -r requirements-optional.txt
```

On Debian, Ubuntu, or WSL, install the base system dependencies and Git LFS:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs libsndfile1
```

Install the PortAudio development package only when using `pyaudio` for
real-time audio:

```bash
sudo apt-get install -y portaudio19-dev
```

## Original checkpoints

The weights under `ckpts/` are original project assets tracked with Git LFS.
After cloning the repository, download them with:

```bash
git lfs install
git lfs pull
```

Use the following commands to check that the files are downloaded objects
rather than unresolved LFS pointer files:

```bash
git lfs status
stat -c '%n %s bytes' ckpts/multi_stem/*.ckpt
```

The repository cleanup did not modify, replace, or regenerate any weight under
`ckpts/`.

## Speaker-guided inference

`inference_with_spk.py` remains an experimental entry point. It runs in three
stages: first it performs a zero-conditioned blind separation, then it invokes
an external script to extract an embedding from the resulting `vocals.wav`,
and finally it performs a guided separation with that embedding.

This workflow requires an additional CAMPPlus model and a speaker-embedding
extraction script. Specify the external Python environment and script with:

```bash
export MSS_SPEAKER_PYTHON=/path/to/speaker-env/bin/python
export MSS_SPEAKER_SCRIPT=/path/to/batch_extract_embeddings.py
```

For each input file, the external script must create:

```text
<store_dir>/embeddings/<input-filename>/embedding.npy
```

The singular filename `embedding.npy` used here is different from the plural
`embeddings.npy` expected inside each training-track directory. Every run
deletes and recreates `<store_dir>/embeddings`, so use a dedicated
`--store_dir` and do not keep files that must be preserved in that
subdirectory.

The two CAMPPlus paths currently recorded in the repository are legacy
gitlinks, but no `.gitmodules` file records their repository URLs. A fresh
clone therefore cannot retrieve that model automatically. Obtain and place the
asset only after confirming its original source; do not overwrite anything
under `ckpts/` with an unknown or substitute weight.

## Training

Training uses Hugging Face Accelerate. After installing the training
dependencies, configure Accelerate for the local machine:

```bash
accelerate config
```

Fill in the model, configuration, training data, validation data, output
directory, and GPU settings near the top of
[`train_accelerate.sh`](train_accelerate.sh), then run the script. See the
English [training data preparation guide](docs/TRAINING_DATA.md) for dataset
layouts and the validation checklist.

Before launching a multi-GPU job, first use one GPU and a small dataset to
verify data loading and one forward/backward pass.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall -q .
bash -n train_accelerate.sh infer_with_spk.sh
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## Known limitations

- The repository does not include a complete end-to-end test dataset. CI runs
  only static checks that do not require model weights.
- The legacy discriminator directory still contains historical interfaces and
  script paths that have not all been migrated to package-relative imports.
- Components such as TensorRT and SageAttention depend heavily on the local
  CUDA toolkit and compiler and are not included in the default dependencies.
- The repository currently has no explicit root-level license. The project
  owner should add licensing terms before the project is used or distributed.

## Acknowledgments

Parts of the training and inference structure were inspired by
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Third-party copyright and license notices retained in individual source files
continue to apply.
