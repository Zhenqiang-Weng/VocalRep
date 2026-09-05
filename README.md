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

The main entry point performs blind separation, extracts the dominant vocal
identity with the official pretrained **CAM++ / CAMPPlus (campp)** encoder, and
runs speaker-guided separation. The encoder uses 16 kHz audio, 80-bin Kaldi
filterbanks with mean normalization, and 192-dimensional embeddings.

The official
[iic/speech_campplus_sv_zh_en_16k-common_advanced](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)
revision `v1.0.0` is downloaded on first use into `checkpoints/campp/`.
Both configuration and weights are SHA-256 verified. No external extraction
script or separate Python environment is required. For offline use, supply
`--spk_model_path /path/to/campp` with `configuration.json` and
`campplus_cn_en_common.pt`. The encoder is frozen during extraction.

Run from the repository root:

```bash
python inference_with_spk.py \
  --model_type spk_bs_roformer \
  --config_path ckpts/multi_stem/config.yaml \
  --start_check_point ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt \
  --input_folder /path/to/mixtures \
  --store_dir results/separation \
  --inference_batch_size 1
```

Alternatively, set `INPUT_FOLDER` and run `bash infer_with_spk.sh`.
Set `SPK_MODEL_PATH` for an offline encoder or `FORCE_CPU=true` for CPU inference.
`--inference_chunk_size` overrides the chunk length in samples; 176400 samples
is a practical four-second setting at 44.1 kHz.

Outputs are organized as:

```text
results/separation/
  wo_spk/<track>/{vocals,backing_vocal,instrumental}.wav
  with_spk/<track>/{vocals,backing_vocal,instrumental}.wav
  embeddings/<track>/embedding.npy
  run_summary.json
```

The embedding has shape `(1, 192)`; training datasets use the separate filename
`embeddings.npy`. Each invocation recomputes its selected tracks and overwrites
their output files, without deleting the embedding directory. Empty input,
invalid checkpoints, non-finite audio, and missing active vocal segments are
reported as errors. Silent vocals do not produce a fabricated speaker identity.

## Reproducible three-stem test

```bash
python -m scripts.smoke_test_three_stem \
  --checkpoint ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt
```

This downloads the project's public Boy Friend demo, verifies asset checksums,
and runs both separation passes. It validates stereo output lengths, sample
rates, finite non-silent stems, and the pretrained embedding. Audio, generated
outputs, and downloaded models are ignored by Git. See
[the test record](docs/TESTING.md) for measured results and scope.

## Training

Prepare CAM++ speaker features for both the aligned training and validation sets:

```bash
python -m scripts.extract_speaker_embeddings \
  --data-path /path/to/train /path/to/valid \
  --device cuda:0
```

The extractor reads `<root>/<track>/vocals.wav` or `vocals.flac` and saves
float32 `(N, 192)` arrays to `<root>/<track>/embeddings.npy`. It uses the pretrained
CAM++ model, skips valid existing arrays, and reports missing/silent vocals as
errors. Use `--device cpu` for CPU extraction, `--model-dir /path/to/campp` for an
offline encoder, or `--overwrite` to regenerate existing features. See the
[speaker embedding guide](docs/TRAINING_DATA.md#speaker-embeddings) for segmentation,
optional dominant-speaker filtering, and failure reports.

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

See [auxiliary model training](docs/AUXILIARY_MODELS.md) for discriminator and
diffusion configuration, loss ownership, checkpoint compatibility, and tests.

## Development checks

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
ruff check .
ruff format --check .
python scripts/check_english_comments.py
python -m pytest -q
python -m compileall -q -x '/(\.venv|checkpoints|test_sample|results)/' .
bash -n train_accelerate.sh infer_with_spk.sh
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## Repository status

This repository is still being organized, and the author will continue to improve and supplement its contents. It is not yet ready for general use or distribution and has not been released under an open-source license. Unless an explicit license is added, all rights are reserved, and no permission is granted to use, copy, modify, or distribute any part of this repository.

## Acknowledgments

Parts of the training and inference structure were copied from
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Third-party copyright and license notices retained in individual source files
continue to apply.
