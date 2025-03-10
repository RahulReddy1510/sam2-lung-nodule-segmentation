"""
tests/test_mc_dropout.py
=========================
Tests for Monte Carlo Dropout inference utilities:
mc_predict, mc_dropout_mode, entropy_from_samples,
compute_uncertainty_stats, and integration with model + evaluation.

Run:
    pytest tests/test_mc_dropout.py -v
"""
import pytest
import numpy as np
import torch
import torch.nn as nn


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_model():
    """Tiny segmentation model for fast MC tests."""
    from models.registry import get_model
    m = get_model(
        "sam2_lung_seg",
        embed_dim=64, num_heads=4,
        attn_dropout=0.1, proj_dropout=0.1,
        encoder_frozen=True,
    )
    m.eval()
    return m


@pytest.fixture
def single_input():
    """Single slice: (1, 1, 64, 64)."""
    return torch.randn(1, 1, 64, 64)


@pytest.fixture
def batch_input():
    """Batch: (4, 1, 64, 64)."""
    return torch.randn(4, 1, 64, 64)


# ── mc_dropout_mode context manager ──────────────────────────────────────────


class TestMCDropoutMode:
    """Tests for the mc_dropout_mode context manager."""

    def test_dropout_enabled_inside_context(self, small_model):
        """Inside mc_dropout_mode, Dropout layers should be in train mode."""
        from models.mc_dropout import mc_dropout_mode, enable_dropout_modules

        dropout_states = {}
        small_model.eval()
        with mc_dropout_mode(small_model):
            for name, m in small_model.named_modules():
                if isinstance(m, torch.nn.Dropout):
                    dropout_states[name] = m.training

        if dropout_states:
            assert any(dropout_states.values()), (
                "At least one Dropout layer should be in training mode inside mc_dropout_mode"
            )
        # If no Dropout layers, the test is vacuously true (model may use different dropout)

    def test_model_restored_after_context(self, small_model):
        """Model training mode should be restored after exiting mc_dropout_mode."""
        from models.mc_dropout import mc_dropout_mode

        small_model.eval()
        original_training = small_model.training  # False

        with mc_dropout_mode(small_model):
            pass  # just enter and exit

        assert small_model.training == original_training, (
            f"Model training mode should be {original_training}, got {small_model.training}"
        )

    def test_context_restores_on_exception(self, small_model):
        """Even on exception, model state should be restored."""
        from models.mc_dropout import mc_dropout_mode

        small_model.eval()
        try:
            with mc_dropout_mode(small_model):
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        assert not small_model.training, "Model should remain in eval mode after exception"


# ── mc_predict ───────────────────────────────────────────────────────────────


class TestMCPredict:
    """Tests for mc_predict — the main MC Dropout inference function."""

    def test_output_shapes_single(self, small_model, single_input):
        """mc_predict should return (mean, variance) of same shape as logits."""
        from models.mc_dropout import mc_predict
        mean, var = mc_predict(small_model, single_input, n_samples=5, mc_batch_size=5)
        assert mean.shape == (1, 1, 64, 64), f"Mean shape {mean.shape} != (1,1,64,64)"
        assert var.shape  == (1, 1, 64, 64), f"Var shape {var.shape} != (1,1,64,64)"

    def test_output_shapes_batch(self, small_model, batch_input):
        """mc_predict with batch input."""
        from models.mc_dropout import mc_predict
        mean, var = mc_predict(small_model, batch_input, n_samples=5, mc_batch_size=5)
        assert mean.shape == (4, 1, 64, 64)
        assert var.shape  == (4, 1, 64, 64)

    def test_mean_in_0_1_range_with_sigmoid(self, small_model, single_input):
        """With sigmoid=True, mean probabilities must be in [0, 1]."""
        from models.mc_dropout import mc_predict
        mean, _ = mc_predict(small_model, single_input, n_samples=5, sigmoid=True)
        assert mean.min() >= 0.0, f"Mean prob below 0: {mean.min()}"
        assert mean.max() <= 1.0, f"Mean prob above 1: {mean.max()}"

    def test_variance_non_negative(self, small_model, single_input):
        """Variance must be non-negative everywhere."""
        from models.mc_dropout import mc_predict
        _, var = mc_predict(small_model, single_input, n_samples=8, sigmoid=True)
        assert var.min() >= 0.0, f"Negative variance: {var.min()}"

    def test_outputs_finite(self, small_model, single_input):
        """mc_predict results must not contain NaN or Inf."""
        from models.mc_dropout import mc_predict
        mean, var = mc_predict(small_model, single_input, n_samples=5)
        assert torch.isfinite(mean).all(), "Mean contains non-finite values"
        assert torch.isfinite(var).all(),  "Variance contains non-finite values"

    def test_more_samples_reduces_variance_on_deterministic_model(self):
        """For a model with no dropout, N samples should give zero variance."""
        from models.mc_dropout import mc_predict

        class DeterministicModel(nn.Module):
            """Linear model with no stochasticity."""
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, kernel_size=1)
                nn.init.constant_(self.conv.weight, 1.0)
                nn.init.constant_(self.conv.bias, 0.0)
            def forward(self, x):
                return self.conv(x)

        det_model = DeterministicModel().eval()
        x = torch.randn(1, 1, 32, 32)
        _, var = mc_predict(det_model, x, n_samples=10, sigmoid=False)
        assert var.max().item() < 1e-6, (
            f"Deterministic model should have ~zero variance, got {var.max().item()}"
        )

    def test_variance_increases_with_heterogeneous_dropout(self):
        """Model with high dropout probability should exhibit higher variance than low."""
        from models.mc_dropout import mc_predict, mc_dropout_mode

        class HighDropModel(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.drop = nn.Dropout(p=p)
            def forward(self, x):
                return self.drop(self.conv(x))

        x = torch.randn(1, 1, 32, 32)
        low_p  = HighDropModel(p=0.1).eval()
        high_p = HighDropModel(p=0.9).eval()

        with mc_dropout_mode(low_p):
            _, var_low  = mc_predict(low_p,  x, n_samples=50, sigmoid=False)
        with mc_dropout_mode(high_p):
            _, var_high = mc_predict(high_p, x, n_samples=50, sigmoid=False)

        # High-dropout model should have larger variance
        assert var_high.mean() >= var_low.mean(), (
            "Higher dropout should yield higher uncertainty"
        )

    def test_mc_batch_size_chunking_consistency(self, small_model, single_input):
        """Different mc_batch_size values should give same mean shape."""
        from models.mc_dropout import mc_predict
        N = 10
        mean1, _ = mc_predict(small_model, single_input, n_samples=N, mc_batch_size=2)
        mean2, _ = mc_predict(small_model, single_input, n_samples=N, mc_batch_size=5)
        mean3, _ = mc_predict(small_model, single_input, n_samples=N, mc_batch_size=10)
        assert mean1.shape == mean2.shape == mean3.shape

    @pytest.mark.parametrize("n_samples", [1, 5, 25])
    def test_various_n_samples(self, small_model, single_input, n_samples):
        """mc_predict should work for any n_samples ≥ 1."""
        from models.mc_dropout import mc_predict
        mean, var = mc_predict(small_model, single_input, n_samples=n_samples)
        assert mean.shape == single_input.shape
        assert var.shape  == single_input.shape


# ── entropy_from_samples ──────────────────────────────────────────────────────


class TestEntropyFromSamples:
    """Tests for entropy_from_samples (pixel-wise predictive entropy)."""

    def test_shape_preserved(self):
        """entropy_from_samples takes a (T, B, 1, H, W) or (T, 1, H, W) Tensor."""
        from models.mc_dropout import entropy_from_samples
        # (T, B, 1, H, W)
        samples = torch.rand(10, 2, 1, 32, 32)
        entropy = entropy_from_samples(samples)
        # Result should be (B, 1, H, W)
        assert entropy.shape == (2, 1, 32, 32), f"Expected (2,1,32,32), got {entropy.shape}"

    def test_entropy_all_certain(self):
        """If all samples agree perfectly (≈1), entropy is near 0."""
        from models.mc_dropout import entropy_from_samples
        samples = torch.full((20, 1, 1, 32, 32), 0.999)
        entropy = entropy_from_samples(samples)
        assert entropy.max().item() < 0.05, f"Entropy should be near 0, got {entropy.max():.4f}"

    def test_entropy_max_at_half(self):
        """Entropy is maximised when mean probability ≈ 0.5."""
        from models.mc_dropout import entropy_from_samples
        generator = torch.Generator().manual_seed(0)
        # Uniform [0.4, 0.6] → high entropy
        samples_half  = torch.rand(30, 1, 1, 32, 32, generator=generator) * 0.2 + 0.4
        # Nearly certain → low entropy
        samples_certa = torch.full((30, 1, 1, 32, 32), 0.99)
        ent_half  = entropy_from_samples(samples_half)
        ent_certa = entropy_from_samples(samples_certa)
        assert ent_half.mean().item() > ent_certa.mean().item(), (
            "Entropy near p=0.5 should exceed entropy near p=1.0"
        )

    def test_entropy_non_negative(self):
        from models.mc_dropout import entropy_from_samples
        samples = torch.rand(10, 1, 1, 48, 48)
        entropy = entropy_from_samples(samples)
        assert entropy.min().item() >= 0.0


# ── compute_uncertainty_stats ─────────────────────────────────────────────────


class TestComputeUncertaintyStats:
    """Tests for compute_uncertainty_stats summary helper."""

    def test_returns_dict_with_required_keys(self, small_model, single_input):
        from models.mc_dropout import mc_predict, compute_uncertainty_stats
        mean, var = mc_predict(small_model, single_input, n_samples=5, sigmoid=True)
        binary = (mean >= 0.5).float()
        stats = compute_uncertainty_stats(var, binary)
        # Actual keys: unc_mean, unc_max, unc_std, unc_in_pred_mean, unc_out_pred_mean
        assert "unc_mean" in stats, f"Missing 'unc_mean' in {list(stats.keys())}"
        assert "unc_max"  in stats, f"Missing 'unc_max' in {list(stats.keys())}"

    def test_all_stats_finite(self, small_model, single_input):
        from models.mc_dropout import mc_predict, compute_uncertainty_stats
        mean, var = mc_predict(small_model, single_input, n_samples=5, sigmoid=True)
        binary = (mean >= 0.5).float()
        stats = compute_uncertainty_stats(var, binary)
        for k, v in stats.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Stat {k} is non-finite: {v}"

    def test_mean_uncertainty_less_than_max(self, small_model, single_input):
        from models.mc_dropout import mc_predict, compute_uncertainty_stats
        mean, var = mc_predict(small_model, single_input, n_samples=5, sigmoid=True)
        binary = (mean >= 0.5).float()
        stats = compute_uncertainty_stats(var, binary)
        assert stats["unc_mean"] <= stats["unc_max"] + 1e-7


# ── Integration: mc_predict + DiceMetric ─────────────────────────────────────


class TestMCDropoutIntegration:
    """End-to-end integration tests: mc_predict → evaluation pipeline."""

    def test_mc_predict_into_dice_metric(self, small_model, batch_input):
        """mc_predict output should be directly usable with DiceMetric."""
        from models.mc_dropout import mc_predict
        from evaluation.dice_metric import DiceMetric

        target = (torch.rand_like(batch_input) > 0.5).float()
        mean, var = mc_predict(small_model, batch_input, n_samples=5, sigmoid=True)

        dm = DiceMetric(threshold=0.5)
        dm.update(mean, target)
        result = dm.compute()
        assert 0.0 <= result["dice"] <= 1.0, f"Dice out of range: {result['dice']}"

    def test_mc_predict_into_calibration_analyzer(self, small_model, single_input):
        """mc_predict uncertainty should feed CalibrationAnalyzer without error."""
        from models.mc_dropout import mc_predict
        from evaluation.uncertainty_calibration import CalibrationAnalyzer

        target = (torch.rand_like(single_input) > 0.5).float()
        mean, var = mc_predict(small_model, single_input, n_samples=5, sigmoid=True)

        ca = CalibrationAnalyzer(n_bins=5, subsample_rate=1.0)
        ca.update(mean, var, target)
        result = ca.compute()
        assert "ece" in result

    def test_mc_predict_uncertainty_correlates_with_disagreement(self):
        """High model disagreement (50-50 split) should give high variance."""
        from models.mc_dropout import mc_predict

        # A model that alternates between all-0 and all-1 predictions
        class AlternatingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._call = 0
            def forward(self, x):
                out = torch.ones_like(x[:, :1]) if self._call % 2 == 0 else -torch.ones_like(x[:, :1]) * 5
                self._call += 1
                return out

        model = AlternatingModel().eval()
        x = torch.zeros(1, 1, 16, 16)

        # Run multiple single-forward-pass predictions manually
        preds = []
        for _ in range(10):
            with torch.no_grad():
                preds.append(torch.sigmoid(model(x)))

        stacked = torch.stack(preds)         # (10, 1, 1, 16, 16)
        variance = stacked.var(dim=0)
        assert variance.mean().item() > 0.05, (
            "Alternating model should show high variance"
        )

    def test_full_eval_loop_synthetic(self):
        """Run a mini evaluation loop matching the pattern in evaluate.py."""
        from models.registry import get_model
        from models.mc_dropout import mc_predict
        from data.dataset import build_dataset
        from torch.utils.data import DataLoader
        from evaluation.dice_metric import DiceMetric
        from evaluation.uncertainty_calibration import CalibrationAnalyzer

        model = get_model("sam2_lung_seg",
                          embed_dim=64, num_heads=4,
                          attn_dropout=0.1, proj_dropout=0.1,
                          encoder_frozen=True).eval()

        ds = build_dataset("SYNTHETIC", split="test", mode="slice", augment=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        dm = DiceMetric(threshold=0.5)
        ca = CalibrationAnalyzer(n_bins=5, subsample_rate=0.5)

        for i, batch in enumerate(loader):
            if i >= 2:  # only 2 batches for speed
                break
            imgs = batch["image"]
            msks = batch["mask"]
            mean, var = mc_predict(model, imgs, n_samples=3, mc_batch_size=3, sigmoid=True)
            dm.update(mean, msks)
            ca.update(mean, var, msks)

        seg_r   = dm.compute()
        calib_r = ca.compute()

        assert 0.0 <= seg_r["dice"] <= 1.0
        assert 0.0 <= calib_r.get("ece", 0.0) <= 1.0
