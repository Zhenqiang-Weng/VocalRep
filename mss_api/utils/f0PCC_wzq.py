# -*- coding: utf-8 -*-
"""
Extract F0 with RMVPE and compute log-F0 PCC

Features:
1. Extract audio F0 using RMVPE
2. Compute log-F0 Pearson correlation between two recordings
3. Support batch processing of audio pairs
4. Output detailed statistics

Dependencies:
    pip install librosa numpy scipy torch soundfile
    Requires RMVPE model weights
"""

import os
import sys
import argparse
import json
import numpy as np
import librosa
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.stats import pearsonr
import soundfile as sf

# Add the project root to the import path
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from modules.rmvpe import RMVPE


def convert_numpy_types(obj):
    """
    Recursively convert NumPy values to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


class F0PCCCalculator:
    """F0 PCC calculator"""

    def __init__(self, model_path: str, device: str = None, is_half: bool = False):
        """
        Initialize the F0 PCC calculator

        Args:
            model_path: Path to the RMVPE model
            device: Compute device ('cuda', 'cpu', 'mps')
            is_half: Whether to use half precision
        """
        self.device = device or self._get_device()
        self.is_half = is_half

        # Initialize the RMVPE model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RMVPE model file does not exist: {model_path}")

        print(f"Loading the RMVPE model: {model_path}")
        print(f"Using device: {self.device}")

        self.rmvpe = RMVPE(model_path=model_path, is_half=is_half, device=self.device)
        print("RMVPE model loaded")

    def _get_device(self) -> str:
        """Select a compute device automatically"""
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def extract_f0(
        self, audio_path: str, target_sr: int = 16000, thred: float = 0.03
    ) -> Optional[np.ndarray]:
        """
        Extract audio F0 using RMVPE

        Args:
            audio_path: Audio file path
            target_sr: Target sample rate
            thred: F0 extraction threshold

        Returns:
            F0 sequence as a NumPy array, or None if extraction fails
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Extract F0
            f0 = self.rmvpe.infer_from_audio(audio, thred=thred)
            return f0

        except Exception as e:
            print(f"[Error] F0 extraction failed {audio_path}: {e}")
            return None

    def calculate_logf0_pcc(self, f0_1: np.ndarray, f0_2: np.ndarray) -> Dict:
        """
        Compute log-F0 PCC between two F0 sequences

        Args:
            f0_1: First F0 sequence
            f0_2: Second F0 sequence

        Returns:
            Dictionary containing PCC results
        """
        # Ensure the sequences have equal length
        min_len = min(len(f0_1), len(f0_2))
        f0_1 = f0_1[:min_len]
        f0_2 = f0_2[:min_len]

        # Find frames that are voiced in both sequences
        voiced_mask_1 = f0_1 > 0
        voiced_mask_2 = f0_2 > 0
        common_voiced_mask = voiced_mask_1 & voiced_mask_2

        if np.sum(common_voiced_mask) < 10:  # Require at least 10 voiced frames
            return {
                "pcc": 0.0,
                "p_value": 1.0,
                "voiced_frames_1": np.sum(voiced_mask_1),
                "voiced_frames_2": np.sum(voiced_mask_2),
                "common_voiced_frames": np.sum(common_voiced_mask),
                "total_frames": min_len,
                "error": "insufficient_voiced_frames",
            }

        # Extract F0 values from jointly voiced frames
        voiced_f0_1 = f0_1[common_voiced_mask]
        voiced_f0_2 = f0_2[common_voiced_mask]

        # Convert to the logarithmic domain
        log_f0_1 = np.log(voiced_f0_1 + 1e-8)  # Add epsilon to avoid log(0)
        log_f0_2 = np.log(voiced_f0_2 + 1e-8)

        # Compute Pearson correlation
        try:
            pcc, p_value = pearsonr(log_f0_1, log_f0_2)

            return {
                "pcc": float(pcc),
                "p_value": float(p_value),
                "voiced_frames_1": int(np.sum(voiced_mask_1)),
                "voiced_frames_2": int(np.sum(voiced_mask_2)),
                "common_voiced_frames": int(np.sum(common_voiced_mask)),
                "total_frames": int(min_len),
                "mean_log_f0_1": float(np.mean(log_f0_1)),
                "mean_log_f0_2": float(np.mean(log_f0_2)),
                "std_log_f0_1": float(np.std(log_f0_1)),
                "std_log_f0_2": float(np.std(log_f0_2)),
            }

        except Exception as e:
            return {
                "pcc": 0.0,
                "p_value": 1.0,
                "voiced_frames_1": int(np.sum(voiced_mask_1)),
                "voiced_frames_2": int(np.sum(voiced_mask_2)),
                "common_voiced_frames": int(np.sum(common_voiced_mask)),
                "total_frames": int(min_len),
                "error": str(e),
            }

    def calculate_pcc_for_pair(self, audio_path_1: str, audio_path_2: str) -> Dict:
        """
        Compute log-F0 PCC for an audio pair

        Args:
            audio_path_1: First audio file path
            audio_path_2: Second audio file path

        Returns:
            Dictionary containing complete results
        """
        result = {"audio_1": audio_path_1, "audio_2": audio_path_2, "success": False}

        # Extract F0
        print(f"Processing: {os.path.basename(audio_path_1)} vs {os.path.basename(audio_path_2)}")

        f0_1 = self.extract_f0(audio_path_1)
        if f0_1 is None:
            result["error"] = f"Cannot extract F0: {audio_path_1}"
            return result

        f0_2 = self.extract_f0(audio_path_2)
        if f0_2 is None:
            result["error"] = f"Cannot extract F0: {audio_path_2}"
            return result

        # Compute PCC
        pcc_result = self.calculate_logf0_pcc(f0_1, f0_2)
        result.update(pcc_result)
        result["success"] = True

        print(f"  logF0 PCC: {pcc_result['pcc']:.4f} (p={pcc_result['p_value']:.4f})")
        print(
            f"  Jointly voiced frames: {pcc_result['common_voiced_frames']}/{pcc_result['total_frames']}"
        )

        return result

    def batch_calculate_pcc(self, audio_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Compute log-F0 PCC for multiple audio pairs

        Args:
            audio_pairs: List of audio file pairs

        Returns:
            Dictionary containing all results
        """
        results = []
        successful_pccs = []

        print(f"Start batch processing {len(audio_pairs)} audio pairs...")

        for i, (audio_1, audio_2) in enumerate(audio_pairs, 1):
            print(f"\n[{i}/{len(audio_pairs)}]", end=" ")

            result = self.calculate_pcc_for_pair(audio_1, audio_2)
            results.append(result)

            if result["success"] and "error" not in result:
                successful_pccs.append(result["pcc"])

        # Compute statistics
        stats = self._calculate_statistics(successful_pccs)

        return {
            "results": results,
            "statistics": stats,
            "total_pairs": len(audio_pairs),
            "successful_pairs": len(successful_pccs),
        }

    def _calculate_statistics(self, pccs: List[float]) -> Dict:
        """Compute PCC statistics"""
        if not pccs:
            return {"count": 0, "error": "no_valid_results"}

        pccs = np.array(pccs)

        return {
            "count": len(pccs),
            "mean": float(np.mean(pccs)),
            "std": float(np.std(pccs)),
            "min": float(np.min(pccs)),
            "max": float(np.max(pccs)),
            "median": float(np.median(pccs)),
            "q25": float(np.percentile(pccs, 25)),
            "q75": float(np.percentile(pccs, 75)),
        }


def load_audio_pairs_from_file(file_path: str) -> List[Tuple[str, str]]:
    """
    Load audio pairs from a file

    Format: two tab-separated audio paths per line
    """
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                print(f"[Warning] Invalid format on line {line_num}; skipping: {line}")
                continue

            audio_1, audio_2 = parts
            if not os.path.exists(audio_1):
                print(f"[Warning] File does not exist: {audio_1}")
                continue
            if not os.path.exists(audio_2):
                print(f"[Warning] File does not exist: {audio_2}")
                continue

            pairs.append((audio_1, audio_2))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract F0 with RMVPE and compute log-F0 PCC")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/rmvpe/rmvpe.pt",
        help="Path to the RMVPE model",
    )
    parser.add_argument("--audio1", type=str, help="First audio file path")
    parser.add_argument("--audio2", type=str, help="Second audio file path")
    parser.add_argument(
        "--pairs_file", type=str, help="Path to a list with two audio paths per line"
    )
    parser.add_argument(
        "--output", type=str, default="f0pcc_results.json", help="Output result file path"
    )
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], help="Compute device")
    parser.add_argument("--half", action="store_true", help="Use half precision")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--threshold", type=float, default=0.03, help="F0 extraction threshold")

    args = parser.parse_args()
    # Validate arguments
    if not args.audio1 and not args.pairs_file:
        parser.error("Specify --audio1 and --audio2, or --pairs_file")

    if args.audio1 and not args.audio2:
        parser.error("--audio2 is required when --audio1 is provided")

    # Initialize the calculator
    calculator = F0PCCCalculator(model_path=args.model_path, device=args.device, is_half=args.half)

    # Prepare audio pairs
    if args.pairs_file:
        print(f"Load audio pairs from file: {args.pairs_file}")
        audio_pairs = load_audio_pairs_from_file(args.pairs_file)
        print(f"Loaded {len(audio_pairs)} audio pairs")
    else:
        audio_pairs = [(args.audio1, args.audio2)]

    if not audio_pairs:
        print("No valid audio pairs; exiting")
        return

    # Compute PCC
    results = calculator.batch_calculate_pcc(audio_pairs)

    # Save results after converting NumPy types
    print(f"\nSaving results to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(results), f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    print(f"Total audio pairs: {results['total_pairs']}")
    print(f"Successfully processed pairs: {results['successful_pairs']}")

    if results["statistics"]["count"] > 0:
        stats = results["statistics"]
        print(f"log-F0 PCC statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Standard deviation: {stats['std']:.4f}")
        print(f"  Minimum: {stats['min']:.4f}")
        print(f"  Maximum: {stats['max']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  25%Percentile: {stats['q25']:.4f}")
        print(f"  75%Percentile: {stats['q75']:.4f}")

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
