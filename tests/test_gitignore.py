"""Verify artifact ignore rules without hiding project source or configuration."""

from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def ignore_repository(tmp_path):
    """Use an isolated Git repository, independent of tracked files and user rules."""
    source = Path(__file__).resolve().parents[1] / ".gitignore"
    (tmp_path / ".gitignore").write_bytes(source.read_bytes())
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    return tmp_path


def ignored_paths(repository, paths):
    """Query hypothetical paths without creating any model or dataset binaries."""
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "--stdin"],
        cwd=repository,
        input="\n".join(paths) + "\n",
        text=True,
        capture_output=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return set(result.stdout.splitlines())


def test_generated_artifacts_are_ignored(ignore_repository):
    paths = [
        "__pycache__/module.cpython-310.pyc",
        ".venv/lib/python3.10/site-packages/example.py",
        ".venv-test/bin/python",
        ".conda/env_vars.json",
        "build/lib/example.py",
        "dist/vocalrep.whl",
        "vocalrep.egg-info/PKG-INFO",
        ".pytest_cache/cache/nodeids",
        ".ipynb_checkpoints/example.ipynb",
        ".coverage.worker1",
        "coverage.xml",
        ".env",
        ".env.production",
        "mss_api/.env.local",
        "secrets.yaml",
        "credentials.json",
        ".hf_token",
        ".wandb/settings",
        "local/private.key",
        "local/certificate.pem",
        ".cursor/rules.md",
        ".claude/settings.json",
        "music/mix.WAV",
        "music/vocals.FlAc",
        "music/reference.M4A",
        "music/reference.AIFF",
        "music/reference.opus",
        "music/reference.mp3",
        "clips/demo.MP4",
        "clips/demo.mkv",
        "song/embeddings.npy",
        "song/features.npz",
        "metadata_4.pkl",
        "mss_api/metadata.pkl",
        "features/dataset.h5",
        "features/shard.parquet",
        "archives/dataset.tar.gz",
        "archives/stems.zip",
        "data/manifest.json",
        "datasets/manifest.csv",
        "train/song_001/manifest.json",
        "valid/song_001/manifest.json",
        "samples/demo_note.txt",
        "separated/run_summary.json",
        "test_sample/fixture/readme.txt",
        "mss_api/data/manifest.json",
        "mss_api/test_sample/fixture.json",
        ".cache/modelscope/hub/configuration.json",
        ".accelerate/default_config.yaml",
        ".torchinductor/cache.json",
        "torchinductor_root/fx_graph.py",
        ".triton/kernel/cache.json",
        "results/run_summary.json",
        "output/run_summary.json",
        "mss_api/results/run_summary.json",
        "mss_api/outputs/metrics.json",
        "mss_api/checkpoints/campp/configuration.json",
        "pretrained_models/campp/configuration.json",
        "weights/config.json",
        "discriminator/logs/metrics.json",
        "wandb/run-123/config.yaml",
        "tb_logs/run_1/events.tfevents",
        "logs/train.log.1",
        "slurm-12345.err",
        "custom/events.out.tfevents.123",
        "custom/last_discriminator_ckpt/state.json",
        "custom/last_additional_model_ckpt/state.json",
        "model_structure_and_precision.txt",
        "torch_compile_debug/run_1/log.txt",
        "profiler/trace.json",
        "profile_run.prof",
        "capture.nsys-rep",
        "kernel.ncu-rep",
        "trace.qdrep",
        "run.trace.json",
        "ckpts/speech_campplus_sv_zh_en_16k-common_advanced/configuration.json",
        "mss_api/ckpt/model.pt",
        "custom/model.pt",
        "custom/model.pth",
        "ckpts/new_model.ckpt",
        "custom/model.safetensors",
        "custom/model.torchscript",
        "custom/model.weights",
        "custom/pytorch_model-00001-of-00002.bin",
        "exported/model.json",
        "mss_api/exported_spk/configuration.json",
        "custom/model.onnx",
        "custom/model.onnx.data",
        "custom/model.trt",
        "custom/model.plan",
        "custom/model.timing.cache",
        "downloads/model.part",
        "temporary.tmp",
        "config.yaml.bak",
        "local.pid",
        "tmp/report.json",
        "Desktop.ini",
    ]
    assert ignored_paths(ignore_repository, paths) == set(paths)


def test_source_config_and_environment_templates_remain_visible(ignore_repository):
    paths = [
        ".gitignore",
        ".gitattributes",
        ".github/workflows/quality.yml",
        ".env.example",
        ".env.sample",
        ".env.template",
        ".env.production.example",
        "mss_api/.env.example",
        "NOTICE.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "train_accelerate.sh",
        "scripts/extract_speaker_embeddings.py",
        "tests/test_speaker_extraction.py",
        "tests/test_gitignore.py",
        "discriminator/data/dataset.py",
        "discriminator/config/default.yaml",
        "diffusion/models/config.py",
        "models/bs_roformer/bs_roformer.py",
        "models/core.py",
        "spk_extract/campplus_exportable.py",
        "mss_api/inference_wrapper.py",
        "utils/train_utils.py",
        "ckpts/multi_stem/config.yaml",
        "ckpts/new_model/configuration.json",
        "ckpts/new_model/tokenizer.json",
        "docs/TRAINING_DATA.md",
        "docs/architecture.svg",
        "configs/train.yaml",
        "scripts/dataset_manifest.csv",
    ]
    assert not ignored_paths(ignore_repository, paths)
