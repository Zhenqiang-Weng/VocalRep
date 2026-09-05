"""Reproduce speaker-guided separation on the project's public Boy Friend demo."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inference_with_spk import proc_folder
from utils.download import download_verified, sha256_file


DEMO_URL = "https://zhenqiang-weng.github.io/VocalRep_/demo/short_records/Boy_Friend"
ASSETS = {
    "mix.WAV": "2f40210cd3de44e041647be2c16f97a720053786e1d220b62d4490622d083422",
    "vocals.wav": "c31ab34cb59db2530d0e6d324e41f8ca913a3ca580d54f9b8ccc8580c0d21687",
    "backing_vocal.wav": "0057dcec3c6fe1165fdeb1efc54907ed0db344209950eeaa4c03ea86c5d6813a",
    "instrumental.wav": "d937e8977d4ac501b6ff7aea9747311fdc4e4d8602a641b2b603b07aace475b4",
}
CHECKPOINT_SHA256 = "423a4b5313b5938a678a91fa753412a08419e3b62593448b85d7d3f0c70a4db9"


def main() -> None:
    """Download the test fixture, run both separation passes, and validate outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("test_sample/boy_friend"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/boy_friend"))
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=176400)
    args = parser.parse_args()
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("This smoke test requires the original project checkpoint.")
    for filename, checksum in ASSETS.items():
        download_verified(f"{DEMO_URL}/{filename}", args.work_dir / "source" / filename, checksum)
    # The upstream mix.WAV contains MP3 data; decode it into a real PCM WAV.
    mixture, sr = sf.read(args.work_dir / "source" / "mix.WAV", always_2d=True, dtype="float32")
    input_dir = args.work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    sf.write(input_dir / "boy_friend.wav", mixture, sr, subtype="FLOAT")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    proc_folder(
        {
            "model_type": "spk_bs_roformer",
            "config_path": "ckpts/multi_stem/config.yaml",
            "start_check_point": str(args.checkpoint),
            "input_folder": str(input_dir),
            "store_dir": str(args.output_dir),
            "force_cpu": args.force_cpu,
            "inference_batch_size": 1,
            "inference_chunk_size": args.chunk_size,
            "disable_detailed_pbar": True,
        }
    )
    report = {
        "source": DEMO_URL,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "duration_seconds": len(mixture) / sr,
        "sample_rate": sr,
        "channels": mixture.shape[1],
        "outputs": {},
        "note": "Public demo stems are model outputs, not ground-truth references; no SDR claim is made.",
    }
    for phase in ("wo_spk", "with_spk"):
        for stem in ("vocals", "backing_vocal", "instrumental"):
            output = args.output_dir / phase / "boy_friend" / f"{stem}.wav"
            audio, output_sr = sf.read(output, always_2d=True, dtype="float32")
            if output_sr != sr or audio.shape != mixture.shape or not np.isfinite(audio).all():
                raise ValueError(f"Invalid output shape, rate, or sample values: {output}")
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if rms <= 1e-8:
                raise ValueError(f"Silent output: {output}")
            report["outputs"][f"{phase}/{stem}"] = {
                "shape": list(audio.shape),
                "rms": rms,
                "peak": float(np.max(np.abs(audio))),
                "sha256": sha256_file(output),
            }
    embedding = np.load(args.output_dir / "embeddings/boy_friend/embedding.npy")
    if embedding.shape != (1, 192) or not np.isfinite(embedding).all():
        raise ValueError("Invalid pretrained CAM++ embedding.")
    report["embedding_shape"] = list(embedding.shape)
    report["device"] = (
        "cpu" if args.force_cpu or not torch.cuda.is_available() else torch.cuda.get_device_name(0)
    )
    report["torch_version"] = torch.__version__
    report_path = args.output_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Three-stem smoke test passed. Report: {report_path}")


if __name__ == "__main__":
    main()
