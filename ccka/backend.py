"""Select the available autodiff backend for simulation."""

from __future__ import annotations

try:
    import jax as _jax
    import optax as _optax
except ImportError:
    try:
        import torch as _torch
    except ImportError as exc:
        raise ImportError(
            "CCKA simulation requires either JAX with Optax or PyTorch. "
            "Install one backend with `pip install .[jax]` or `pip install .[torch]`."
        ) from exc
    BACKEND = "torch"
else:
    BACKEND = "jax"


__all__ = ["BACKEND"]
