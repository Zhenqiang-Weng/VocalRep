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
        "local/private.key",
        "local/certificate.pem",
        "music/mix.WAV",
        "music/vocals.FlAc",
        "music/reference.M4A",
        "music/reference.AIFF",
        "music/reference.opus",
        "music/reference.mp3",
        "song/embeddings.npy",
        "song/features.npz",
        "metadata_4.pkl",
        "mss_api/metadata.pkl",
        "data/manifest.json",
        "datasets/manifest.csv",
        "test_sample/fixture/readme.txt",
        "mss_api/data/manifest.json",
        "mss_api/test_sample/fixture.json",
        ".cache/modelscope/hub/configuration.json",
        ".triton/kernel/cache.json",
        "results/run_summary.json",
        "mss_api/results/run_summary.json",
        "mss_api/outputs/metrics.json",
        "mss_api/checkpoints/campp/configuration.json",
        "discriminator/logs/metrics.json",
        "wandb/run-123/config.yaml",
        "logs/train.log.1",
        "custom/events.out.tfevents.123",
        "custom/last_discriminator_ckpt/state.json",
        "custom/model.pt",
        "custom/model.pth",
        "ckpts/new_model.ckpt",
        "custom/model.safetensors",
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
        "mss_api/models/bs_roformer/bs_roformer.py",
        "ckpts/multi_stem/config.yaml",
        "ckpts/new_model/configuration.json",
        "docs/TRAINING_DATA.md",
        "docs/architecture.svg",
        "configs/train.yaml",
        "scripts/dataset_manifest.csv",
    ]
    assert not ignored_paths(ignore_repository, paths)
