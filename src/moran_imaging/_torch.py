"""Pytorch utilities."""

import torch


def get_backend() -> str:
    """Returns the appropriate backend based on availability."""
    if torch.cuda.is_available():
        backend = "cuda"
    elif torch.backends.mps.is_available():
        backend = "mps"
    else:
        backend = "cpu"
    return backend


def to_backend(model):
    """Creates instance of the model with appropriate backend."""
    backend = get_backend()
    if hasattr(model, "to"):
        return model.to(backend)

    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    return model
