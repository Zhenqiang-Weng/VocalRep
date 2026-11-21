# VocalRep: Structure-Aware Vocal Representations for Multimodal Generation

Official PyTorch implementation of **VocalRep: Structure-Aware Vocal Representations for Multimodal Generation**.

**Project Page:** https://zhenqiang-weng.github.io/VocalRep_/

VocalRep is a structure-aware learning framework for disentangling **lead vocals, backing harmonies, and instrumental components** while maintaining role consistency across long-form audio. The framework combines global vocal identity conditioning with ranking-based objectives to obtain temporally stable and semantically unambiguous vocal representations for downstream multimodal generation tasks, including singing voice conversion and audio-driven lip synchronization.

This repository contains the source-separation models, Accelerate training runtime, speaker-guided inference pipeline, and auxiliary discriminator/diffusion components associated with VocalRep. Supported separation backbones include BS-Roformer, Mel-Band Roformer, SCNet, and BandIt v2.

> [!IMPORTANT]
> This repository is still being organized as an experimental research codebase.
> The basic inference pipeline, speaker embedding extractor, and unified
> Accelerate training runtime are available, while some experimental backbones,
> TensorRT export utilities, and auxiliary training configurations remain under
> active validation. See [`NOTICE.md`](NOTICE.md) for rights, usage boundaries,
> and third-party attributions.

## Repository layout

```text
VocalRep/
├── ckpts/multi_stem/                         # Original Git LFS checkpoint and config.yaml
├── models/
│   ├── bs_roformer/                          # BS-Roformer, Mel-Band Roformer, exportable variants, SDPA/Sage attention
│   ├── bandit_v2/                            # BandIt v2 separation backbone
│   ├── scnet/                                # SCNet separation backbones
│   └── preprocess/                           # BF16-safe STFT/iSTFT and multi-resolution STFT loss
├── spk_extract/                              # Pretrained CAM++ loader, clustering, and ONNX/TRT export helper
├── discriminator/                            # Mel, HiFi-GAN wave, and music GAN discriminators
├── diffusion/                                # Flow-matching DiT/Transformer/MLP enhancement modules
├── utils/
│   ├── train_utils.py                        # Shared Accelerate training, validation, checkpointing, and VRAM/speed runtime
│   ├── training_cli.py                       # Unified training CLI argument parser
│   ├── dataset.py / dataset_with_spk.py      # Multi-layout audio and speaker-conditioned datasets
│   └── model_utils.py / settings.py / ...    # Batched demixing, optimizer builders, metrics, losses, RMVPE
├── mss_api/                                  # Inference wrapper and TensorRT export scripts (reusing root modules)
├── scripts/                                  # Speaker embedding extraction, data validation, static checks, utilities
├── tests/                                    # CPU regression tests and .gitignore verification
├── docs/                                     # Data prep, auxiliary training, and validation documentation
├── train_accelerate_bf16.py                  # Speaker-guided Accelerate training entry point
├── train_accelerate_bf16_with_discriminator.py # Speaker-guided + GAN discriminator training entry point
├── train_accelerate_discriminator_diffusion.py # General separator + discriminator + diffusion training entry point
├── train_accelerate.sh                       # Validated multi-GPU Accelerate training launcher
├── inference_with_spk.py / infer_with_spk.sh # Two-pass blind + speaker-guided inference entry points
├── NOTICE.md                                 # Rights reservation, responsibilities, disclaimer, third-party notices
├── SECURITY.md                               # Vulnerability reporting and safe checkpoint/credential guidelines
└── CONTRIBUTING.md                           # Contribution workflow, roles, and code/LFS governance
```

## Environment setup

Recommended environment:

- Linux or WSL2
- Python 3.10+
- An NVIDIA GPU (a CPU is sufficient for basic checks, but inference and
  training with large models will be slow)
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

For programmatic API usage or ONNX/TensorRT export helpers, see
[`mss_api/inference_wrapper.py`](mss_api/inference_wrapper.py),
[`mss_api/export_to_tensorrt.py`](mss_api/export_to_tensorrt.py), and
[`spk_extract/campplus_exportable.py`](spk_extract/campplus_exportable.py).

## Reproducible three-stem test

```bash
python -m scripts.smoke_test_three_stem \
  --checkpoint ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt
```

This downloads the project's public Boy Friend demo, verifies asset checksums,
and runs both separation passes. It validates stereo output lengths, sample
rates, finite non-silent stems, and the pretrained embedding. Audio, generated
outputs, and downloaded models are ignored by Git. See
[the validation record](docs/TESTING.md) for measured results and scope.

## Training

### 1. Prepare speaker embeddings and validate datasets

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
[training data preparation guide](docs/TRAINING_DATA.md) for dataset layouts,
speaker embedding options, and the pre-training validation checklist.

### 2. Configure and launch Accelerate training

Training uses Hugging Face Accelerate. After installing `requirements-train.txt`,
configure Accelerate for the local machine:

```bash
accelerate config
```

Edit the paths, model selection, and runtime flags near the top of
[`train_accelerate.sh`](train_accelerate.sh) and launch training with
`bash train_accelerate.sh`. Available entry points:

- [`train_accelerate_bf16.py`](train_accelerate_bf16.py) — speaker-guided separation (supports exportable and standard Roformers).
- [`train_accelerate_bf16_with_discriminator.py`](train_accelerate_bf16_with_discriminator.py) — speaker-guided separation with per-stem GAN discriminators.
- [`train_accelerate_discriminator_diffusion.py`](train_accelerate_discriminator_diffusion.py) — general separator with optional discriminator and flow-matching diffusion enhancement.

### 3. Speed and VRAM optimization controls

All three Accelerate entry points share [`utils/train_utils.py`](utils/train_utils.py)
and [`utils/training_cli.py`](utils/training_cli.py), providing:

| Option / `train_accelerate.sh` knob | Purpose |
| --- | --- |
| `--gradient-checkpointing` / `GRADIENT_CHECKPOINTING` | Enable or disable activation checkpointing across Roformer layers (including exportable variants during training only) to lower peak VRAM |
| `--gradient-accumulation-steps` / `GRADIENT_ACCUMULATION_STEPS` | Micro-step gradient accumulation (kept at `1` when auxiliary discriminator/diffusion wrappers manage their own `grad_acc_step`) |
| `--compile` (`--compile-mode`) / `COMPILE_MODEL` | Optional `torch.compile` acceleration (`default`, `reduce-overhead`, `max-autotune`) |
| `--allow-tf32` / `ALLOW_TF32` | Enable TF32 matmul/cuDNN on Ampere+ GPUs (automatically disabled under `--deterministic`) |
| `--persistent-workers`, `--prefetch-factor` | Keep DataLoader workers warm across epochs and prefetch batches |
| `--log-interval`, `--valid-every`, `--skip-valid`, `--empty-cache-interval` | Reduce host-GPU synchronization overhead and control validation / cache-clear cadence |

On CUDA GPUs, fused `Adam`/`AdamW`, `zero_grad(set_to_none=True)`, cached STFT
windows, BF16-safe FP32 FFT/iSTFT, CPU-accumulated batched validation, and
micro-step diffusion backpropagation are enabled automatically while keeping
existing checkpoint `state_dict` keys unchanged.

Before launching a multi-GPU job, first use one GPU and a small dataset to
verify data loading and one forward/backward pass. See
[auxiliary model training](docs/AUXILIARY_MODELS.md) for discriminator and
diffusion configuration, loss ownership, and checkpoint compatibility.

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

## Rights, responsibilities, and governance

- **Repository status & rights:** This repository is still being organized, and
  the author will continue to improve and supplement its contents. It is not yet
  ready for general use or distribution and has not been released under an
  open-source license. Unless an explicit license is added, all rights are
  reserved, and no permission is granted to use, copy, modify, or distribute any
  part of this repository.
- **Rights, disclaimer & third-party notices:** See [`NOTICE.md`](NOTICE.md) for
  maintainer/user/contributor responsibilities, ethical and audio copyright
  compliance, warranty disclaimer, limitation of liability, and the full list of
  third-party attributions.
- **Security & safe usage:** See [`SECURITY.md`](SECURITY.md) for private
  vulnerability reporting and checkpoint/credential safety practices.
- **Contributing:** See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution
  workflow, roles, code standards, and Git LFS rules.

## Acknowledgments

Parts of the training and inference structure were copied from
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Third-party copyright and license notices retained in individual source files
continue to apply; see [`NOTICE.md`](NOTICE.md) for the consolidated attribution
summary.
