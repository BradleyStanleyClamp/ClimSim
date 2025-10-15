"""
Utility functions for selecting and managing models. The key component is we use a single config + model name for dealing with different models, e.g loading, training, optimizing different architectures.
"""


def get_model_config(cfg, model_name: str):
    """
    Gets the model config file from the larger config file, and checks its validity.
    """
    if model_name not in cfg.models:
        raise ValueError(f"Model name {model_name} not found in config file.")
    model_cfg = cfg.models[model_name]
    return model_cfg