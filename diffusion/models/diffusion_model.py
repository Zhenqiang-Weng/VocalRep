"""
Diffusion model registry and default implementation
Model classes are managed by `diffusion.models.registry`, with implementations in separate files.
"""

from .registry import register_diffusion_model, get_diffusion_model, list_diffusion_models


# Convenience function to create the default model
def create_default_model(config):
    """Create the default diffusion model"""
    model_type = getattr(config, "model_type", "transformer")
    return get_diffusion_model(model_type, config)
