"""
Model registry for SAM2LungSegmentor variants.

Provides a simple key-value registry that maps string names to model
configurations, enabling config-file driven model selection.

Usage::

    from models.registry import get_model, ModelRegistry

    # Build the default full model
    model = get_model("sam2_lung_seg")

    # Register a custom variant
    @ModelRegistry.register("my_variant")
    def build_my_variant(**kwargs):
        return build_model(embed_dim=128, num_heads=4, **kwargs)

    # Or use the class API
    ModelRegistry.register_config("compact", embed_dim=128, num_heads=4)
    model = get_model("compact")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from models.sam2_finetune import SAM2LungSegmentor, build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Simple string-keyed registry for model factory functions.

    Each registered entry is a callable that accepts arbitrary kwargs and
    returns a ``SAM2LungSegmentor`` (or subclass). This allows config files to
    specify a model variant by name without hard-coding class references.

    Class attributes
    ----------------
    _registry : dict
        Maps variant name → factory callable.
    """

    _registry: Dict[str, Callable[..., SAM2LungSegmentor]] = {}

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator that registers a model factory under the given name.

        Parameters
        ----------
        name : str
            Registry key (must be unique).

        Returns
        -------
        Callable
            The original factory function, unchanged.

        Raises
        ------
        KeyError
            If ``name`` is already registered.

        Examples
        --------
        ::

            @ModelRegistry.register("my_model")
            def build_my_model(**kwargs):
                return build_model(embed_dim=128, **kwargs)
        """

        def decorator(fn: Callable) -> Callable:
            if name in cls._registry:
                raise KeyError(
                    f"ModelRegistry: '{name}' is already registered. "
                    "Use a different name or call unregister() first."
                )
            cls._registry[name] = fn
            logger.debug("ModelRegistry: registered '%s' → %s", name, fn.__name__)
            return fn

        return decorator

    @classmethod
    def register_config(
        cls,
        name: str,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        encoder_frozen: bool = True,
        **kwargs: Any,
    ) -> None:
        """Register a model variant from a simple config dict.

        Parameters
        ----------
        name : str
            Registry key.
        sam2_checkpoint : str or None
            Path to SAM2 checkpoint. None → FallbackEncoder.
        sam2_config : str or None
            SAM2 config name.
        embed_dim : int
            Feature channel dimension. Default 256.
        num_heads : int
            Attention heads. Default 8.
        attn_dropout : float
            Attention dropout probability. Default 0.1.
        proj_dropout : float
            FFN dropout probability. Default 0.1.
        encoder_frozen : bool
            Freeze encoder at init. Default True.
        **kwargs
            Additional kwargs forwarded to ``build_model``.
        """
        config = dict(
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            encoder_frozen=encoder_frozen,
            **kwargs,
        )

        def _factory(**override_kwargs: Any) -> SAM2LungSegmentor:
            merged = dict(config)
            merged.update(override_kwargs)
            return build_model(**merged)

        _factory.__name__ = f"build_{name}"
        if name in cls._registry:
            logger.warning("ModelRegistry: overwriting existing entry '%s'", name)
        cls._registry[name] = _factory
        logger.info(
            "ModelRegistry: registered config '%s'  embed_dim=%d", name, embed_dim
        )

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registered variant.

        Parameters
        ----------
        name : str
            Registry key to remove.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"ModelRegistry: '{name}' not found.")
        del cls._registry[name]
        logger.debug("ModelRegistry: unregistered '%s'", name)

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> SAM2LungSegmentor:
        """Build a model by registry name.

        Parameters
        ----------
        name : str
            Registry key.
        **kwargs
            Override kwargs forwarded to the factory function.

        Returns
        -------
        SAM2LungSegmentor
            A fully constructed model.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"ModelRegistry: '{name}' not registered. " f"Available: {cls.list()}"
            )
        factory = cls._registry[name]
        logger.info("ModelRegistry: building '%s'", name)
        return factory(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        """Return a sorted list of all registered variant names.

        Returns
        -------
        list of str
            All registered keys.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable summary of all registered variants.

        Returns
        -------
        str
            Multi-line summary string.
        """
        if not cls._registry:
            return "ModelRegistry: (empty)"
        lines = ["ModelRegistry variants:"]
        for name, fn in sorted(cls._registry.items()):
            lines.append(f"  {name!r:<30} → {fn.__name__}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Register built-in model variants
# ---------------------------------------------------------------------------

# Default production model (FallbackEncoder — no SAM2 required)
ModelRegistry.register_config(
    name="sam2_lung_seg",
    embed_dim=256,
    num_heads=8,
    attn_dropout=0.10,
    proj_dropout=0.10,
    encoder_frozen=True,
)

# Ablation: compact model for fast development / CI testing
ModelRegistry.register_config(
    name="compact",
    embed_dim=128,
    num_heads=4,
    attn_dropout=0.10,
    proj_dropout=0.10,
    encoder_frozen=False,
)

# Ablation: higher dropout (from ablation config higher_mc_dropout)
ModelRegistry.register_config(
    name="high_dropout",
    embed_dim=256,
    num_heads=8,
    attn_dropout=0.20,
    proj_dropout=0.20,
    encoder_frozen=True,
)

# Ablation: frozen encoder for the full training run
ModelRegistry.register_config(
    name="frozen_encoder_full",
    embed_dim=256,
    num_heads=8,
    attn_dropout=0.10,
    proj_dropout=0.10,
    encoder_frozen=True,  # Never unfrozen in training schedule
)


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------


def get_model(name: str = "sam2_lung_seg", **kwargs: Any) -> SAM2LungSegmentor:
    """Build a registered model by name.

    Thin wrapper around ``ModelRegistry.build``.

    Parameters
    ----------
    name : str
        Model variant name. Default ``"sam2_lung_seg"``.
    **kwargs
        Override kwargs merged into the factory call.

    Returns
    -------
    SAM2LungSegmentor
        Constructed model.

    Examples
    --------
    >>> model = get_model("sam2_lung_seg")
    >>> model = get_model("compact", encoder_frozen=False)
    """
    return ModelRegistry.build(name, **kwargs)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 65)
    print("ModelRegistry — demo")
    print("=" * 65)

    print(f"\n{ModelRegistry.summary()}\n")

    for variant in ModelRegistry.list():
        model = get_model(variant)
        x = torch.randn(1, 1, 64, 64)
        logits = model(x)
        n_params = model.num_parameters(trainable_only=False)
        print(
            f"  [{variant}]  output={tuple(logits.shape)}  "
            f"params={n_params:,}  SAM2={'yes' if model._using_sam2 else 'no'}"
        )
        assert logits.shape == (1, 1, 64, 64)

    print("\nAll registry variants passed forward-pass check. ✓")

    # Test register / unregister
    @ModelRegistry.register("test_variant")
    def _build_test(**kwargs):
        return build_model(embed_dim=64, num_heads=2, **kwargs)

    assert "test_variant" in ModelRegistry.list()
    ModelRegistry.unregister("test_variant")
    assert "test_variant" not in ModelRegistry.list()
    print("register / unregister cycle: ✓")
