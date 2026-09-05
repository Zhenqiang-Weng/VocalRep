"""Lazy model registration with explicit configuration propagation."""

import importlib
import inspect

from .config import DiffusionConfig


DIFFUSION_MODEL_REGISTRY = {}
_BUILTINS = {
    "dit": "diffusion.models.dit.dit",
    "transformer": "diffusion.models.transformer.transformer_diffusion",
    "simple_mlp": "diffusion.models.simple_mlp.simple_mlp",
    "template": "diffusion.models.template.template_diffuser",
}


def register_diffusion_model(name: str):
    """Register a model class and reject accidental duplicate names."""

    def decorator(cls):
        key = name.lower()
        if key in DIFFUSION_MODEL_REGISTRY and DIFFUSION_MODEL_REGISTRY[key] is not cls:
            raise ValueError(f"Diffusion model is already registered: {name}")
        DIFFUSION_MODEL_REGISTRY[key] = cls
        return cls

    return decorator


def get_diffusion_model(name: str, config: DiffusionConfig | None = None):
    """Instantiate a selected model with its configuration and model keyword arguments."""
    name = name.lower()
    if name not in DIFFUSION_MODEL_REGISTRY and name in _BUILTINS:
        importlib.import_module(_BUILTINS[name])
    if name not in DIFFUSION_MODEL_REGISTRY:
        raise ValueError(f"Unknown diffusion model: {name}. Available: {list_diffusion_models()}")
    config = config or DiffusionConfig(model_type=name)
    cls = DIFFUSION_MODEL_REGISTRY[name]
    kwargs = dict(config.model_kwargs)
    if "config" in inspect.signature(cls).parameters:
        return cls(config=config, **kwargs)
    return cls(**kwargs)


def list_diffusion_models() -> list[str]:
    """List built-in and explicitly registered models without importing them."""
    return sorted(set(_BUILTINS) | set(DIFFUSION_MODEL_REGISTRY))
