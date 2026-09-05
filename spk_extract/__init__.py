"""
CAMP++ Speaker Verification Model
Complete package with model definition, checkpoint management, and initialization

Usage:
    # Quick load from checkpoint
    >>> from campplus import load_model
    >>> model = load_model('./checkpoint')

    # Create and save checkpoint
    >>> from campplus import CAMPPlus, save_checkpoint
    >>> model = CAMPPlus(feat_dim=80, embedding_size=512)
    >>> save_checkpoint(model, './checkpoint')

    # Advanced usage
    >>> from campplus import init_from_checkpoint, create_checkpoint
    >>> model = init_from_checkpoint('./checkpoint', device='cuda')
"""

import os
import json
import torch
import torchaudio.compliance.kaldi as Kaldi
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, NamedTuple
import scipy.signal
from sklearn.cluster import SpectralClustering
from collections import Counter

# ============================================================================
# Import Core Model Components
# ============================================================================

from .campplus_model import (
    CAMPPlus,
    FCM,
    TDNNLayer,
    CAMLayer,
    CAMDenseTDNNLayer,
    CAMDenseTDNNBlock,
    TransitLayer,
    DenseLayer,
    StatsPool,
    BasicResBlock,
    get_nonlinear,
    statistics_pooling,
    create_campplus_model,
)

# ============================================================================
# Checkpoint Management
# ============================================================================


class CheckpointManager:
    """Manages model checkpoints with configuration files"""

    CONFIG_FILE = "configuration.json"
    MODEL_FILE = "campplus_cn_en_common.pt"

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(
        self, model: torch.nn.Module, config: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """Save model checkpoint with configuration"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = self.MODEL_FILE

        # Save model state dict
        model_path = self.checkpoint_dir / filename
        torch.save(model.state_dict(), model_path)

        # Save configuration
        config_path = self.checkpoint_dir / self.CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Config saved to: {config_path}")
        return str(model_path)

    def load(self, filename: Optional[str] = None, device: str = "cpu") -> tuple:
        """Load model checkpoint and configuration"""
        if filename is None:
            filename = self.MODEL_FILE

        # Load configuration
        config_path = self.checkpoint_dir / self.CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration not found: {config_path}\n"
                f"Expected checkpoint structure:\n"
                f"  {self.checkpoint_dir}/\n"
                f"  ├── configuration.json\n"
                f"  └── {filename}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load model state dict
        model_path = self.checkpoint_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        print(f"✓ Loaded from: {self.checkpoint_dir}")
        return state_dict, config

    def exists(self) -> bool:
        """Check if checkpoint exists"""
        config_exists = (self.checkpoint_dir / self.CONFIG_FILE).exists()
        model_exists = (self.checkpoint_dir / self.MODEL_FILE).exists()
        return config_exists and model_exists


# ============================================================================
# Configuration Management
# ============================================================================


def create_config(
    feat_dim: int = 80,
    embedding_size: int = 512,
    growth_rate: int = 32,
    bn_size: int = 4,
    init_channels: int = 128,
    config_str: str = "batchnorm-relu",
    memory_efficient: bool = True,
    output_level: str = "segment",
    **kwargs,
) -> Dict[str, Any]:
    """
    Create model configuration dictionary

    Args:
        feat_dim: Input feature dimension (default: 80)
        embedding_size:  Output embedding dimension (default: 512)
        growth_rate: Dense block growth rate (default: 32)
        bn_size: Bottleneck size multiplier (default: 4)
        init_channels: Initial TDNN channels (default: 128)
        config_str: Activation configuration (default: 'batchnorm-relu')
        memory_efficient: Use gradient checkpointing (default: True)
        output_level: 'segment' or 'frame' (default: 'segment')
        **kwargs: Additional metadata

    Returns:
        Configuration dictionary
    """
    config = {
        "model_type": "campplus",
        "model_version": "1.0",
        "framework": "pytorch",
        "model_config": {
            "feat_dim": feat_dim,
            "embedding_size": embedding_size,
            "growth_rate": growth_rate,
            "bn_size": bn_size,
            "init_channels": init_channels,
            "config_str": config_str,
            "memory_efficient": memory_efficient,
            "output_level": output_level,
        },
        "feature_config": {
            "sample_rate": kwargs.get("sample_rate", 16000),
            "feature_type": "fbank",
            "num_mel_bins": feat_dim,
            "frame_length": kwargs.get("frame_length", 25),
            "frame_shift": kwargs.get("frame_shift", 10),
        },
        "training_info": {
            "trained_on": kwargs.get("trained_on", "unknown"),
            "num_speakers": kwargs.get("num_speakers", -1),
            "num_epochs": kwargs.get("num_epochs", -1),
            "description": kwargs.get("description", ""),
        },
    }

    return config


# ============================================================================
# Main Initialization Functions
# ============================================================================


def init_from_checkpoint(
    model_dir: str,
    device: str = "cpu",
    pretrained_model: str = "campplus_cn_en_common.pt",
    strict: bool = True,
) -> CAMPPlus:
    """
    Initialize CAMP++ model from checkpoint directory

    This is the main initialization function that mimics ModelScope's pattern.
    It loads both the configuration and model weights from a checkpoint directory.

    Args:
        model_dir: Directory containing configuration. json and model checkpoint
        device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
        pretrained_model: Checkpoint filename (default: 'campplus_cn_en_common.pth')
        strict: Strict mode for loading state dict (default: True)

    Returns:
        Initialized and loaded CAMPPlus model

    Example:
        >>> model = init_from_checkpoint('./my_checkpoint', device='cuda')
        >>> features = torch.randn(1, 200, 80)
        >>> embedding = model(features)

    Raises:
        FileNotFoundError: If checkpoint directory or files don't exist
        RuntimeError: If model loading fails
    """
    # Load checkpoint
    manager = CheckpointManager(model_dir)
    state_dict, config = manager.load(filename=pretrained_model, device=device)

    # Extract model config - handle different config structures
    model_config = config.get("model_config", {})

    # If model_config is a string (path to yaml), look for nested config
    if isinstance(model_config, str) or not model_config:
        # Try nested structure: config["model"]["model_config"]
        nested_config = config.get("model", {}).get("model_config", {})
        if nested_config and isinstance(nested_config, dict):
            # Map the nested config keys to CAMPPlus expected keys
            model_config = {
                "feat_dim": nested_config.get("fbank_dim", 80),
                "embedding_size": nested_config.get("emb_size", 192),
            }
        else:
            raise ValueError(
                f"Invalid configuration file. Missing valid 'model_config' section.\n"
                f"Config keys found: {list(config.keys())}"
            )

    # Create model
    try:
        model = CAMPPlus(**model_config)
    except Exception as e:
        raise RuntimeError(f"Failed to create model with config {model_config}: {e}")

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")

    # Setup model
    model.to(device)
    model.eval()

    # Print info
    feat_dim = model_config.get("feat_dim", "unknown")
    emb_size = model_config.get("embedding_size", "unknown")
    print(f"✓ CAMP++ Model Ready")
    print(f"  - Feature dim: {feat_dim}")
    print(f"  - Embedding size: {emb_size}")
    print(f"  - Device: {device}")

    return model


def save_checkpoint(
    model: CAMPPlus,
    checkpoint_dir: str,
    config: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    **config_kwargs,
) -> str:
    """
    Save CAMP++ model to checkpoint directory

    Args:
        model: CAMPPlus model instance to save
        checkpoint_dir: Directory to save checkpoint
        config: Optional configuration dict (auto-generated if None)
        filename: Optional checkpoint filename (default: 'campplus.pth')
        **config_kwargs: Config parameters if auto-generating config

    Returns:
        Path to saved checkpoint file

    Example:
        >>> model = CAMPPlus(feat_dim=80, embedding_size=512)
        >>> save_checkpoint(model, './my_checkpoint',
        ...                 trained_on='VoxCeleb', num_epochs=100)
    """
    manager = CheckpointManager(checkpoint_dir)

    # Auto-generate config if not provided
    if config is None:
        # Try to infer parameters from model
        try:
            feat_dim = config_kwargs.get("feat_dim", 80)
            embedding_size = config_kwargs.get("embedding_size", 512)
            config = create_config(
                feat_dim=feat_dim, embedding_size=embedding_size, **config_kwargs
            )
        except Exception as e:
            raise ValueError(f"Failed to create config: {e}")

    return manager.save(model, config, filename=filename)


def create_checkpoint(
    checkpoint_dir: str,
    feat_dim: int = 80,
    embedding_size: int = 512,
    random_init: bool = True,
    **kwargs,
) -> CAMPPlus:
    """
    Create a new CAMP++ model and save as checkpoint

    Args:
        checkpoint_dir: Directory to save checkpoint
        feat_dim: Input feature dimension
        embedding_size:  Output embedding dimension
        random_init: Use random initialization (default: True)
        **kwargs: Additional model/config parameters

    Returns:
        Created model instance

    Example:
        >>> model = create_checkpoint('./new_checkpoint', feat_dim=80,
        ...                          embedding_size=512)
    """
    # Create model
    model_kwargs = {
        "feat_dim": feat_dim,
        "embedding_size": embedding_size,
        "growth_rate": kwargs.pop("growth_rate", 32),
        "bn_size": kwargs.pop("bn_size", 4),
        "init_channels": kwargs.pop("init_channels", 128),
        "config_str": kwargs.pop("config_str", "batchnorm-relu"),
        "memory_efficient": kwargs.pop("memory_efficient", True),
        "output_level": kwargs.pop("output_level", "segment"),
    }

    model = CAMPPlus(**model_kwargs)

    # Save checkpoint
    config = create_config(**model_kwargs, **kwargs)
    manager = CheckpointManager(checkpoint_dir)
    manager.save(model, config)

    print(f"✓ New checkpoint created at: {checkpoint_dir}")
    return model


# ============================================================================
# Convenient Loading Functions
# ============================================================================


def load_model(checkpoint_dir: str, device: Optional[str] = None, **kwargs) -> CAMPPlus:
    """
    Simple one-line model loading (auto device detection)

    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load on (auto-detect if None)
        **kwargs: Additional arguments for init_from_checkpoint

    Returns:
        Loaded CAMPPlus model

    Example:
        >>> model = load_model('./checkpoint')  # Auto device
        >>> model = load_model('./checkpoint', device='cuda')  # Specific device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return init_from_checkpoint(checkpoint_dir, device=device, **kwargs)


def load_model_cpu(checkpoint_dir: str, **kwargs) -> CAMPPlus:
    """Load model on CPU"""
    return init_from_checkpoint(checkpoint_dir, device="cpu", **kwargs)


def load_model_cuda(checkpoint_dir: str, gpu_id: int = 0, **kwargs) -> CAMPPlus:
    """Load model on CUDA"""
    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"
    return init_from_checkpoint(checkpoint_dir, device=device, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================


def check_checkpoint(checkpoint_dir: str) -> bool:
    """
    Check if a valid checkpoint exists

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        True if valid checkpoint exists, False otherwise
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.exists()


def get_checkpoint_info(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Get checkpoint configuration information

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Configuration dictionary

    Example:
        >>> info = get_checkpoint_info('./checkpoint')
        >>> print(f"Embedding size: {info['model_config']['embedding_size']}")
    """
    config_path = Path(checkpoint_dir) / "configuration.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Audio Loading (soundfile version - ModelScope style)
# ============================================================================


def load_audio_soundfile(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio using soundfile (ModelScope implementation)

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)

    Returns:
        audio: Audio waveform numpy array (samples,)
        sr: Sample rate

    Example:
        >>> audio, sr = load_audio_soundfile('speaker1.wav')
        >>> print(f"Audio shape: {audio.shape}, SR: {sr}")
    """
    # Read audio file
    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert stereo to mono (take first channel)
    if len(audio.shape) == 2:
        audio = audio[:, 0]

    # Resample if needed
    if sr != target_sr:
        import scipy.signal

        audio = scipy.signal.resample_poly(audio, target_sr, sr).astype("float32")
        sr = target_sr

    return audio, sr


# ============================================================================
# Feature Extraction (Fbank)
# ============================================================================


def extract_fbank(
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    apply_cmvn: bool = True,
) -> torch.Tensor:
    """
    Extract Fbank (Filter Bank) features from audio
    Feature extraction used by CAMP++ in ModelScope

    Args:
        audio: Audio waveform (samples,) - numpy array or torch tensor
        sample_rate:  Sample rate (default: 16000 Hz)
        num_mel_bins: Number of mel filter banks (default: 80)
        frame_length: Frame length in milliseconds (default: 25.0)
        frame_shift:  Frame shift in milliseconds (default:  10.0)
        apply_cmvn: Apply Cepstral Mean Normalization (default: True)

    Returns:
        features: Fbank features, shape (time_steps, num_mel_bins)

    Example:
        >>> audio, sr = load_audio_soundfile('test.wav')
        >>> features = extract_fbank(audio, sample_rate=sr)
        >>> print(f"Features shape: {features.shape}")  # (T, 80)
    """
    # Convert numpy to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    # Ensure 2D shape:  (1, samples) for Kaldi. fbank
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    # Extract Fbank features using Kaldi
    # This is the EXACT method used in CAMP++ model
    features = Kaldi.fbank(
        audio,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        frame_length=frame_length,
        frame_shift=frame_shift,
    )

    # Apply Cepstral Mean Normalization (CMVN)
    # Subtract the mean, matching CAMP++ normalization
    if apply_cmvn:
        features = features - features.mean(dim=0, keepdim=True)

    return features


# ============================================================================
# Complete Pipeline Function
# ============================================================================


def load_and_extract_features(
    audio_path: str, target_sr: int = 16000, num_mel_bins: int = 80, apply_cmvn: bool = True
) -> torch.Tensor:
    """
    Load an audio file and extract features in one step

    Args:
        audio_path:  Path to audio file
        target_sr: Target sample rate
        num_mel_bins: Number of mel bins
        apply_cmvn:  Apply mean normalization

    Returns:
        features: Fbank features (time_steps, num_mel_bins)

    Example:
        >>> features = load_and_extract_features('audio. wav')
        >>> print(features.shape)  # torch.Size([T, 80])
    """
    # Step 1: Load audio
    audio, sr = load_audio_soundfile(audio_path, target_sr=target_sr)

    # Step 2: Extract features
    features = extract_fbank(
        audio, sample_rate=sr, num_mel_bins=num_mel_bins, apply_cmvn=apply_cmvn
    )

    return features


def extract_embedding(
    model: CAMPPlus, features: Union[torch.Tensor, Any], device: Optional[str] = None
) -> torch.Tensor:
    """
    Extract speaker embedding from features

    Args:
        model: CAMPPlus model
        features: Input features (batch, time, feat_dim) or convertible to tensor
        device: Device for computation (use model's device if None)

    Returns:
        Speaker embeddings (batch, embedding_size)

    Example:
        >>> model = load_model('./checkpoint')
        >>> features = torch.randn(2, 200, 80)
        >>> embeddings = extract_embedding(model, features)
        >>> print(embeddings.shape)  # (2, 512)
    """
    model.eval()

    # Convert to tensor if needed
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)

    # Ensure correct shape
    if len(features.shape) == 2:
        features = features.unsqueeze(0)

    assert len(features.shape) == 3, f"Expected shape (batch, time, feat_dim), got {features.shape}"

    # Get device
    if device is None:
        device = next(model.parameters()).device

    # Extract embedding
    with torch.no_grad():
        features = features.to(device)
        embeddings = model(features)

    return embeddings.cpu()


# ============================================================================
# Core Similarity Computation (ModelScope Implementation)
# ============================================================================


def compute_cos_similarity(
    emb1: Union[torch.Tensor, np.ndarray], emb2: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute cosine similarity between two embeddings

    Implementation from ModelScope speaker verification pipeline:
    - Uses torch.nn.CosineSimilarity with dim=1 and eps=1e-6
    - Returns a single float value representing similarity in range [-1, 1]

    Mathematical Formula:
        cos_sim = (emb1 · emb2) / (||emb1|| * ||emb2||)

    Args:
        emb1: First embedding, shape (embedding_dim,) or (1, embedding_dim)
        emb2: Second embedding, shape (embedding_dim,) or (1, embedding_dim)

    Returns:
        Cosine similarity score in range [-1, 1]
        - 1.0: Identical embeddings
        - 0.0: Orthogonal embeddings
        - -1.0: Opposite embeddings

    Example:
        >>> emb1 = torch.randn(512)
        >>> emb2 = torch.randn(512)
        >>> score = compute_cos_similarity(emb1, emb2)
        >>> print(f"Similarity: {score:.4f}")
    """
    # Convert numpy to tensor if needed
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)

    # Ensure 2D shape:  (batch_size, embedding_dim)
    if len(emb1.shape) == 1:
        emb1 = emb1.unsqueeze(0)
    if len(emb2.shape) == 1:
        emb2 = emb2.unsqueeze(0)

    # Validate shapes
    assert len(emb1.shape) == 2 and len(emb2.shape) == 2, (
        f"Expected 2D tensors, got shapes {emb1.shape} and {emb2.shape}"
    )

    # Create cosine similarity function
    # dim=1: compute similarity along embedding dimension
    # eps=1e-6: small value to avoid division by zero
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # Compute cosine similarity
    cosine = cos(emb1, emb2)

    # Return scalar value
    return cosine.item()


# ============================================================================
# Dominant Speaker Embedding Extraction with Spectral Clustering
# ============================================================================


class ClusteringResult(NamedTuple):
    """Spectral clustering result"""

    mean_embedding: torch.Tensor  # Mean embedding of the largest cluster
    all_embeddings: np.ndarray  # Embeddings from all segments
    labels: np.ndarray  # Cluster label for each segment
    largest_cluster_label: int  # Label of the largest cluster
    segment_indices: List[
        Tuple[int, int]
    ]  # (start, end) indices of valid segments in the resampled waveform
    n_clusters: int  # Number of clusters


def extract_dominant_speaker_embedding_with_clusters(
    model: CAMPPlus,
    audio: np.ndarray,
    source_sr: int = 44100,
    target_sr: int = 16000,
    segment_duration: float = 2.0,
    energy_threshold: float = 0.1,
    max_clusters: int = 3,
    device: Optional[str] = None,
    debug: bool = False,
    batch_size: int = 8,
    max_segments: int = 64,
) -> Optional[ClusteringResult]:
    """Extract and cluster pretrained speaker embeddings from active audio segments.

    Accept mono or stereo audio in either channel layout. Short final segments
    are repeated to the model window length after energy filtering. Bound both
    inference batches and the number of sampled segments for long recordings.
    Return None for empty or silent audio; reject non-finite samples.
    """
    if source_sr <= 0 or target_sr != 16000:
        raise ValueError("CAM++ requires a positive source rate and a 16000 Hz target rate.")
    if not np.isfinite(segment_duration) or segment_duration < 0.1:
        raise ValueError("Speaker segments must be at least 0.1 seconds long.")
    if batch_size < 1 or max_segments < 1 or max_clusters < 1:
        raise ValueError("Batch size, segment limit, and cluster count must be positive.")
    if not np.isfinite(energy_threshold) or energy_threshold < 0:
        raise ValueError("Energy threshold must be finite and non-negative.")
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[0] in (1, 2):
            audio = audio.mean(axis=0)
        elif audio.shape[1] in (1, 2):
            audio = audio.mean(axis=1)
        else:
            raise ValueError("Expected mono or stereo audio.")
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Audio must be a finite mono or stereo waveform.")
    if audio.size == 0:
        return None
    if source_sr != target_sr:
        audio = scipy.signal.resample_poly(audio, target_sr, source_sr).astype(np.float32)

    segment_samples = int(segment_duration * target_sr)
    segment_indices = []
    for start in range(0, len(audio), segment_samples):
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]
        # At least 100 ms of active audio is needed for a useful embedding.
        if len(segment) >= target_sr // 10 and np.sqrt(np.mean(segment**2)) > energy_threshold:
            segment_indices.append((start, end))
    if not segment_indices:
        return None
    if len(segment_indices) > max_segments:
        selected = np.linspace(0, len(segment_indices) - 1, max_segments, dtype=int)
        segment_indices = [segment_indices[index] for index in selected]

    batches = []
    with torch.inference_mode():
        for offset in range(0, len(segment_indices), batch_size):
            features = []
            for start, end in segment_indices[offset : offset + batch_size]:
                segment = audio[start:end]
                if len(segment) < segment_samples:
                    segment = np.tile(segment, int(np.ceil(segment_samples / len(segment))))
                    segment = segment[:segment_samples]
                features.append(extract_fbank(segment, sample_rate=target_sr))
            embeddings = model(torch.stack(features).to(device))
            batches.append(embeddings.float().cpu().numpy())
    embeddings_np = np.concatenate(batches)
    if not np.isfinite(embeddings_np).all():
        raise ValueError("CAM++ produced non-finite embeddings.")

    # Spectral decomposition requires fewer clusters than samples.
    n_clusters = min(max_clusters, max(1, len(segment_indices) - 1))
    if n_clusters == 1:
        labels = np.zeros(len(segment_indices), dtype=int)
    else:
        norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True).clip(min=1e-8)
        normalized = embeddings_np / norm
        affinity = np.clip((normalized @ normalized.T + 1) / 2, 0, 1)
        labels = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=0,
        ).fit_predict(affinity)
    largest_cluster_label = Counter(labels).most_common(1)[0][0]
    mean_embedding = torch.from_numpy(
        embeddings_np[labels == largest_cluster_label].mean(axis=0)
    ).float()
    if debug:
        print(
            f"CAM++: {len(segment_indices)} segments, {n_clusters} clusters, "
            f"dominant cluster {largest_cluster_label}"
        )
    return ClusteringResult(
        mean_embedding=mean_embedding,
        all_embeddings=embeddings_np,
        labels=labels,
        largest_cluster_label=int(largest_cluster_label),
        segment_indices=segment_indices,
        n_clusters=n_clusters,
    )


def estimate_k_by_eigengap(affinity_matrix, k_max=20, eps=1e-12):
    W = np.asarray(affinity_matrix, dtype=np.float64)
    n = W.shape[0]
    assert W.shape[0] == W.shape[1], "affinity_matrix must be square"

    W = 0.5 * (W + W.T)
    W = np.maximum(W, 0.0)

    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    S = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]

    L = np.eye(n) - S

    m = min(n, k_max + 1)
    evals = np.linalg.eigvalsh(L)[:m]  # Sorted in ascending order
    gaps = np.diff(evals)  # gaps[i] = evals[i+1]-evals[i]

    k = int(np.argmax(gaps[: m - 1]) + 1)  # +1 because i -> k=i+1
    return k, evals, gaps


# Delegate dominant speaker extraction to the clustering function
def extract_dominant_speaker_embedding(
    model: CAMPPlus,
    audio: np.ndarray,
    source_sr: int = 44100,
    target_sr: int = 16000,
    segment_duration: float = 2.0,
    energy_threshold: float = 0.1,
    max_clusters: int = 3,
    device: Optional[str] = None,
    debug: bool = False,
) -> Optional[torch.Tensor]:
    """
    Extract the dominant speaker embedding from 44.1 kHz audio

    Return only the mean embedding of the largest cluster.
    Use extract_dominant_speaker_embedding_with_clusters for complete clustering details
    """
    result = extract_dominant_speaker_embedding_with_clusters(
        model=model,
        audio=audio,
        source_sr=source_sr,
        target_sr=target_sr,
        segment_duration=segment_duration,
        energy_threshold=energy_threshold,
        max_clusters=max_clusters,
        device=device,
        debug=debug,
    )

    if result is None:
        return None

    return result.mean_embedding


# ============================================================================
# Package Information
# ============================================================================

__version__ = "1.0.0"
__author__ = "CAMP++ Implementation"
__all__ = [
    # Main functions
    "init_from_checkpoint",
    "save_checkpoint",
    "create_checkpoint",
    "load_model",
    "load_model_cpu",
    "load_model_cuda",
    # Model classes
    "CAMPPlus",
    "create_campplus_model",
    # Utilities
    "check_checkpoint",
    "get_checkpoint_info",
    "extract_embedding",
    "create_config",
    "compute_cos_similarity",
    "load_and_extract_features",
    "load_audio_soundfile",
    "extract_fbank",
    "extract_dominant_speaker_embedding",
    "extract_dominant_speaker_embedding_with_clusters",  # Added
    "ClusteringResult",  # Added
    # Components (for advanced users)
    "FCM",
    "TDNNLayer",
    "CAMLayer",
    "CAMDenseTDNNLayer",
    "CAMDenseTDNNBlock",
    "TransitLayer",
    "DenseLayer",
    "StatsPool",
    "BasicResBlock",
    "get_nonlinear",
    "statistics_pooling",
    # Checkpoint manager
    "CheckpointManager",
]


# ============================================================================
# Module-level convenience
# ============================================================================


def info():
    """Print package information"""
    print("=" * 70)
    print("CAMP++ Speaker Verification Model")
    print("=" * 70)
    print(f"Version: {__version__}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print("=" * 70)
    print("\nQuick Start:")
    print("  from campplus import load_model")
    print("  model = load_model('./checkpoint')")
    print("\nDocumentation:")
    print("  help(load_model)")
    print("  help(save_checkpoint)")
    print("=" * 70)


# Show info on import if in interactive mode
if __name__ != "__main__":
    import sys

    if hasattr(sys, "ps1"):  # Interactive mode
        info()
