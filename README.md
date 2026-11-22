# VocalRep: Structure-Aware Vocal Representations for Multimodal Generation

Official PyTorch implementation of **VocalRep: Structure-Aware Vocal Representations for Multimodal Generation**.

**Project Page:** https://zhenqiang-weng.github.io/VocalRep_/

VocalRep disentangles **lead vocals, backing harmonies, and instrumental stems** using global vocal identity conditioning (CAM++) and structure-aware objectives. Supported separation backbones include BS-Roformer, Mel-Band Roformer, SCNet, and BandIt v2.

> [!IMPORTANT]
> Experimental research codebase under active organization. See [`NOTICE.md`](NOTICE.md) for rights, usage boundaries, and third-party attributions.

## Setup

Requires Linux/WSL2, Python 3.10+, FFmpeg, Git LFS, and an NVIDIA GPU (CPU works for basic checks only).

```bash
conda create -n mss python=3.10 -y && conda activate mss
# Install PyTorch matching your CUDA version (example: CUDA 13.0)
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
# Optional: training or extra dependencies
python -m pip install -r requirements-train.txt
python -m pip install -r requirements-optional.txt
```

Download the original Git LFS checkpoint under `ckpts/`:

```bash
git lfs install && git lfs pull
```

## Inference

The speaker-guided pipeline runs blind separation, extracts the dominant vocal embedding with pretrained **CAM++** (downloaded on demand into `checkpoints/campp/` with SHA-256 verification), and performs guided separation:

```bash
python inference_with_spk.py \
  --model_type spk_bs_roformer \
  --config_path ckpts/multi_stem/config.yaml \
  --start_check_point ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt \
  --input_folder /path/to/mixtures \
  --store_dir results/separation \
  --inference_batch_size 1
```

Or configure `INPUT_FOLDER` and run `bash infer_with_spk.sh` (`SPK_MODEL_PATH` for offline CAM++, `FORCE_CPU=true` for CPU). Outputs are written to `results/separation/{wo_spk,with_spk,embeddings}` with `run_summary.json`.

Reproducible three-stem smoke test (see [`docs/TESTING.md`](docs/TESTING.md)):

```bash
python -m scripts.smoke_test_three_stem \
  --checkpoint ckpts/multi_stem/model_spk_bs_roformer_ep_5_sisdr_9.8275.ckpt
```

## Training

1. Extract CAM++ speaker embeddings (`embeddings.npy`, float32 `(N, 192)`) for training and validation tracks:

```bash
python -m scripts.extract_speaker_embeddings \
  --data-path /path/to/train /path/to/valid \
  --device cuda:0
```

2. Configure Accelerate (`accelerate config`), edit paths/settings at the top of [`train_accelerate.sh`](train_accelerate.sh), and launch:

```bash
bash train_accelerate.sh
```

- **Entry points** (sharing [`utils/train_utils.py`](utils/train_utils.py)): [`train_accelerate_bf16.py`](train_accelerate_bf16.py), [`train_accelerate_bf16_with_discriminator.py`](train_accelerate_bf16_with_discriminator.py), and [`train_accelerate_discriminator_diffusion.py`](train_accelerate_discriminator_diffusion.py).
- **Speed & VRAM options:** `--gradient-checkpointing`, `--gradient-accumulation-steps`, `--compile`, `--allow-tf32`, `--persistent-workers`, `--prefetch-factor`, `--log-interval`, `--valid-every`, `--skip-valid` (fused CUDA `Adam`/`AdamW`, TF32, cached STFT windows, and BF16-safe FP32 FFT/iSTFT are enabled automatically).
- **Guides:** [`docs/TRAINING_DATA.md`](docs/TRAINING_DATA.md) (dataset layouts & validation) and [`docs/AUXILIARY_MODELS.md`](docs/AUXILIARY_MODELS.md) (discriminator & diffusion training).

## Development Checks

```bash
ruff check . && ruff format --check .
python scripts/check_english_comments.py
python -m pytest -q
python -m compileall -q -x '/(\.venv|checkpoints|test_sample|results)/' .
bash -n train_accelerate.sh infer_with_spk.sh
```

## Repository Status & Governance

This repository is still being organized, and the author will continue to improve and supplement its contents. It is not yet ready for general use or distribution and has not been released under an open-source license. Unless an explicit license is added, all rights are reserved, and no permission is granted to use, copy, modify, or distribute any part of this repository.

- Rights, responsibilities, disclaimer & third-party notices: [`NOTICE.md`](NOTICE.md)
- Security & safe usage policy: [`SECURITY.md`](SECURITY.md)
- Contribution guidelines: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Acknowledgments

Parts of the training and inference structure were copied from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). Third-party copyright and license notices retained in individual source files continue to apply (see [`NOTICE.md`](NOTICE.md)).
