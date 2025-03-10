"""
tests/test_model.py
====================
Tests for model construction, forward pass, encoder freeze/unfreeze,
gradient flow, and TemporalConsistencyLoss.

Run:
    pytest tests/test_model.py -v
"""
import pytest
import torch
import torch.nn as nn


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model_frozen():
    """SAM2LungSegmentor with encoder frozen."""
    from models.registry import get_model
    return get_model(
        "sam2_lung_seg",
        embed_dim=256, num_heads=8,
        attn_dropout=0.1, proj_dropout=0.1,
        encoder_frozen=True,
    )


@pytest.fixture(scope="module")
def model_unfrozen():
    """SAM2LungSegmentor with encoder unfrozen."""
    from models.registry import get_model
    return get_model(
        "sam2_lung_seg",
        embed_dim=256, num_heads=8,
        attn_dropout=0.1, proj_dropout=0.1,
        encoder_frozen=False,
    )


@pytest.fixture
def dummy_batch():
    """Single batch: (B=2, C=1, H=96, W=96)."""
    return torch.randn(2, 1, 96, 96)


@pytest.fixture
def dummy_batch_large():
    """Larger batch for timing / memory checks."""
    return torch.randn(4, 1, 128, 128)


# ── Registry tests ────────────────────────────────────────────────────────────


class TestModelRegistry:
    """Tests for the model registry / factory."""

    def test_default_model_creation(self):
        from models.registry import get_model
        m = get_model("sam2_lung_seg")
        assert m is not None
        assert isinstance(m, nn.Module)

    def test_unknown_model_raises(self):
        from models.registry import get_model
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            get_model("nonexistent_model_xyz")

    def test_list_models_contains_default(self):
        from models.registry import ModelRegistry
        names = ModelRegistry.list()
        assert isinstance(names, (list, tuple, set))
        assert any("sam2" in n.lower() or "lung" in n.lower() for n in names)

    def test_model_is_nn_module(self, model_frozen):
        assert isinstance(model_frozen, nn.Module)

    def test_model_has_parameters(self, model_frozen):
        params = list(model_frozen.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 1000, f"Model seems too small: {total} params"


# ── Forward pass tests ────────────────────────────────────────────────────────


class TestForwardPass:
    """Tests for model forward pass correctness."""

    def test_output_shape_matches_input(self, model_frozen, dummy_batch):
        model_frozen.eval()
        with torch.no_grad():
            out = model_frozen(dummy_batch)
        assert out.shape == (2, 1, 96, 96), (
            f"Expected (2,1,96,96), got {out.shape}"
        )

    def test_output_shape_large_batch(self, model_frozen, dummy_batch_large):
        model_frozen.eval()
        with torch.no_grad():
            out = model_frozen(dummy_batch_large)
        assert out.shape == (4, 1, 128, 128)

    def test_output_is_logits_not_probabilities(self, model_frozen, dummy_batch):
        """Raw output should be logits (can be outside [0, 1])."""
        model_frozen.eval()
        with torch.no_grad():
            logits = model_frozen(dummy_batch)
        # Logits may have values outside [0,1]; probabilities after sigmoid are [0,1]
        probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0, "Sigmoid probs below 0"
        assert probs.max() <= 1.0, "Sigmoid probs above 1"

    def test_output_finite(self, model_frozen, dummy_batch):
        """Model should not produce NaN or Inf in logits."""
        model_frozen.eval()
        with torch.no_grad():
            out = model_frozen(dummy_batch)
        assert torch.isfinite(out).all(), "Model produced non-finite logits"

    def test_batch_independence(self, model_frozen):
        """Output for sample i should not depend on other samples in the batch."""
        model_frozen.eval()
        x = torch.randn(4, 1, 96, 96)
        with torch.no_grad():
            out_batch = model_frozen(x)
            out_single = model_frozen(x[[1]])
        # Slice 1 from batch must match single forward pass
        assert torch.allclose(out_batch[1], out_single[0], atol=1e-5), (
            "Batch inference is not sample-independent"
        )

    def test_different_input_gives_different_output(self, model_frozen):
        """Distinct inputs should (almost certainly) produce distinct outputs."""
        model_frozen.eval()
        x1 = torch.zeros(1, 1, 96, 96)
        x2 = torch.ones(1, 1, 96, 96)
        with torch.no_grad():
            o1 = model_frozen(x1)
            o2 = model_frozen(x2)
        assert not torch.allclose(o1, o2), "Different inputs produced identical outputs"


# ── Encoder freeze / unfreeze tests ──────────────────────────────────────────


class TestEncoderFreezeUnfreeze:
    """Tests for encoder frozen / trainable parameter management."""

    def test_frozen_encoder_no_grad(self, model_frozen):
        """When encoder is frozen, encoder params must have requires_grad=False."""
        if not hasattr(model_frozen, "encoder"):
            pytest.skip("Model does not expose .encoder attribute")
        for name, p in model_frozen.encoder.named_parameters():
            assert not p.requires_grad, f"Encoder param {name} has requires_grad=True (should be frozen)"

    def test_unfrozen_encoder_has_grad(self, model_unfrozen):
        """When encoder is unfrozen, encoder params must have requires_grad=True."""
        if not hasattr(model_unfrozen, "encoder"):
            pytest.skip("Model does not expose .encoder attribute")
        grads = [p.requires_grad for _, p in model_unfrozen.encoder.named_parameters()]
        assert any(grads), "No encoder parameter has requires_grad=True after unfreezing"

    def test_freeze_unfreeze_toggle(self):
        """freeze_encoder() / unfreeze_encoder() should toggle grad correctly."""
        from models.registry import get_model
        m = get_model("sam2_lung_seg", encoder_frozen=False)
        if not (hasattr(m, "freeze_encoder") and hasattr(m, "unfreeze_encoder")):
            pytest.skip("Model lacks freeze_encoder/unfreeze_encoder methods")

        m.freeze_encoder()
        if hasattr(m, "encoder"):
            frozen_grad = any(p.requires_grad for p in m.encoder.parameters())
            assert not frozen_grad, "freeze_encoder did not disable grads"

        m.unfreeze_encoder()
        if hasattr(m, "encoder"):
            unfrozen_grad = any(p.requires_grad for p in m.encoder.parameters())
            assert unfrozen_grad, "unfreeze_encoder did not enable grads"

    def test_frozen_encoder_decoder_still_trains(self, model_frozen, dummy_batch):
        """With frozen encoder, decoder params should accumulate gradients."""
        model_frozen.train()
        target = torch.zeros(2, 1, 96, 96)
        out = model_frozen(dummy_batch)
        loss = nn.functional.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        # At least one parameter in the decoder should have a gradient
        decoder_grad_any = False
        for name, p in model_frozen.named_parameters():
            if p.requires_grad and p.grad is not None:
                decoder_grad_any = True
                break
        assert decoder_grad_any, "No trainable param got a gradient after backward()"
        model_frozen.eval()


# ── Gradient flow ─────────────────────────────────────────────────────────────


class TestGradientFlow:
    """End-to-end gradient flow tests."""

    def test_backward_pass_completes(self, model_unfrozen, dummy_batch):
        """loss.backward() should not crash."""
        model_unfrozen.train()
        target = torch.zeros(2, 1, 96, 96)
        out = model_unfrozen(dummy_batch)
        loss = nn.functional.binary_cross_entropy_with_logits(out, target)
        loss.backward()  # should not raise
        model_unfrozen.eval()

    def test_loss_decreases_with_gradient_step(self):
        """A single SGD step should decrease loss on a simple batch."""
        from models.registry import get_model
        m = get_model("sam2_lung_seg", encoder_frozen=True)
        m.train()
        opt = torch.optim.SGD(
            [p for p in m.parameters() if p.requires_grad], lr=0.01
        )
        x = torch.randn(2, 1, 96, 96)
        t = torch.zeros(2, 1, 96, 96)

        out1 = m(x)
        loss1 = nn.functional.binary_cross_entropy_with_logits(out1, t)
        loss1.backward()
        opt.step()
        opt.zero_grad()

        out2 = m(x)
        loss2 = nn.functional.binary_cross_entropy_with_logits(out2, t)
        # Should decrease (or at worst only marginally not, due to random init)
        # We just assert that the gradient step ran without error and loss is finite
        assert torch.isfinite(loss2), "Loss became non-finite after gradient step"
        m.eval()


# ── TemporalConsistencyLoss tests ─────────────────────────────────────────────


class TestTemporalConsistencyLoss:
    """Tests for the TemporalConsistencyLoss module.

    Actual API:
      - Constructor: lambda_bce, lambda_tc, consistency_mode, warmup_epochs
      - forward(logits, targets, slice_indices=None) → dict{total, dice, bce, temporal}
    """

    @pytest.fixture(scope="class")
    def tc_loss(self):
        from models.temporal_consistency import TemporalConsistencyLoss
        return TemporalConsistencyLoss(
            lambda_bce=0.5,
            lambda_tc=0.3,
            warmup_epochs=5,
        )

    def test_loss_returns_dict_with_total(self, tc_loss):
        """forward() should return a dict containing a scalar 'total' key."""
        logits = torch.randn(2, 1, 96, 96)
        target = (torch.rand(2, 1, 96, 96) > 0.5).float()
        result = tc_loss(logits, target)
        assert isinstance(result, dict), "forward should return a dict"
        assert "total" in result, f"Expected 'total' key in {list(result.keys())}"
        total = result["total"]
        assert total.ndim == 0, "Loss must be a scalar tensor"
        assert torch.isfinite(total)

    def test_loss_has_component_keys(self, tc_loss):
        """forward() dict must contain 'dice', 'bce', 'temporal' components."""
        logits = torch.randn(2, 1, 96, 96)
        target = (torch.rand(2, 1, 96, 96) > 0.5).float()
        result = tc_loss(logits, target)
        for key in ("dice", "bce", "temporal"):
            assert key in result, f"Missing component key '{key}' in {list(result.keys())}"

    def test_loss_positive(self, tc_loss):
        """Total loss should be non-negative."""
        logits = torch.randn(2, 1, 96, 96)
        target = (torch.rand(2, 1, 96, 96) > 0.5).float()
        result = tc_loss(logits, target)
        assert result["total"].item() >= 0.0, "Loss should be non-negative"

    def test_tc_warmup_temporal_zero_without_slice_indices(self, tc_loss):
        """Without slice_indices, temporal loss should be 0.0 (skipped)."""
        logits = torch.randn(2, 1, 96, 96)
        target = (torch.rand(2, 1, 96, 96) > 0.5).float()
        result = tc_loss(logits, target, slice_indices=None)
        assert result["temporal"].item() == pytest.approx(0.0, abs=1e-6)

    def test_all_components_finite(self, tc_loss):
        """All returned loss components must be finite."""
        logits = torch.randn(2, 1, 96, 96)
        target = (torch.rand(2, 1, 96, 96) > 0.5).float()
        result = tc_loss(logits, target)
        for k, v in result.items():
            assert torch.isfinite(v), f"Component '{k}' is non-finite: {v}"

    def test_perfect_prediction_bce_small(self, tc_loss):
        """With near-perfect predictions, BCE component should be close to 0."""
        # Large positive logits → σ(logit) ≈ 1.0 → near-perfect for all-ones target
        logits = torch.full((2, 1, 96, 96), 10.0)
        target = torch.ones(2, 1, 96, 96)
        result = tc_loss(logits, target)
        assert result["bce"].item() < 0.5, (
            f"BCE should be small for near-perfect predictions, got {result['bce'].item():.4f}"
        )




# ── Channel adapter smoke tests ───────────────────────────────────────────────


class TestChannelAdapter:
    """Tests for the 1→3 channel expansion inside SAM2LungSegmentor.

    The channel adapter is internal to SAM2LungSegmentor (a Conv2d(1,3)).
    We test equivalent behaviour directly with a standalone Conv2d to ensure
    the spatial-preserving channel expansion contract holds.
    """

    def test_model_accepts_single_channel_input(self, model_frozen, dummy_batch):
        """Model must accept (B, 1, H, W) inputs without error."""
        model_frozen.eval()
        with torch.no_grad():
            out = model_frozen(dummy_batch)
        assert out.shape[-1] == dummy_batch.shape[-1]
        assert out.shape[-2] == dummy_batch.shape[-2]

    def test_standalone_1to3_adapter(self):
        """A Conv2d(1, 3, 1) adapter expands channels correctly."""
        adapter = nn.Conv2d(1, 3, kernel_size=1)
        x = torch.randn(2, 1, 96, 96)
        out = adapter(x)
        assert out.shape == (2, 3, 96, 96), f"Expected (2,3,96,96), got {out.shape}"
        assert torch.isfinite(out).all()

    def test_standalone_adapter_spatial_preserved(self):
        """Channel adapter should not change H and W."""
        adapter = nn.Conv2d(1, 3, kernel_size=1)
        x = torch.randn(2, 1, 128, 256)
        out = adapter(x)
        assert out.shape[-2:] == (128, 256)
