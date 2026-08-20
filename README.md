# VocalRep: Structure-Aware Vocal Representations for Multimodal Generation

Official PyTorch implementation of **VocalRep: Structure-Aware Vocal Representations for Multimodal Generation**.

**Project Page:** https://zhenqiang-weng.github.io/VocalRep_/

VocalRep is a structure-aware learning framework for disentangling **lead vocals, backing harmonies, and instrumental components** while maintaining role consistency across long-form audio. The framework combines global vocal identity conditioning with ranking-based objectives to obtain temporally stable and semantically unambiguous vocal representations for downstream multimodal generation tasks, including singing voice conversion and audio-driven lip synchronization.

This repository contains the source-separation models, training and inference utilities, and experimental components associated with VocalRep. Supported separation backbones include BS-Roformer, Mel-Band Roformer, SCNet, and BandIt v2.

> [!IMPORTANT]
> This repository is still being organized as an experimental research codebase.
> The basic inference pipeline and several model implementations are available,
> while some training entry points, speaker-model assets, and legacy
> discriminator components are still undergoing end-to-end validation.
> Additional code, checkpoints, and documentation will be released progressively.

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

## Repository status

This repository is still being organized and is not yet ready for general use or distribution. It has not been released under an open-source license. Unless an explicit license is added, all rights are reserved, and no permission is granted to use, copy, modify, or distribute any part of this repository.


## Acknowledgments

Parts of the training and inference structure were copied from
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Third-party copyright and license notices retained in individual source files
continue to apply.
