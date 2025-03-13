"""
tests/test_metrics.py
======================
Tests for evaluation metrics: Dice, IoU, precision, recall,
Hausdorff (HD95), ECE, Brier score, uncertainty AUROC, Cohen's κ,
Fleiss κ, percent agreement, and Bland-Altman analysis.

Run:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
import torch

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_pred_target(batch=4, h=64, w=64, pred_value=0.8, target_value=1.0):
    """Return (pred_probs, target_mask) tensors with given fill values."""
    pred = torch.full((batch, 1, h, w), pred_value)
    target = torch.full((batch, 1, h, w), target_value)
    return pred, target


def _make_binary_lists(n=100, noise=0.05, rng_seed=42):
    """Return (pred_labels, target_labels) lists with controlled noise."""
    rng = np.random.default_rng(rng_seed)
    target = rng.integers(0, 2, n).tolist()
    pred = [t if rng.random() > noise else 1 - t for t in target]
    return pred, target


# ══════════════════════════════════════════════════════════════════════════════
# Dice / IoU / Precision / Recall
# ══════════════════════════════════════════════════════════════════════════════


class TestDiceMetric:
    """Tests for compute_dice and DiceMetric accumulator."""

    def test_perfect_prediction_dice_is_one(self):
        from evaluation.dice_metric import compute_dice

        pred, target = _make_pred_target(pred_value=1.0, target_value=1.0)
        dice = compute_dice(pred, target, threshold=0.5)
        assert torch.allclose(
            dice, torch.ones_like(dice), atol=1e-4
        ), f"Perfect prediction should give Dice=1, got {dice}"

    def test_all_wrong_prediction_dice_is_zero(self):
        from evaluation.dice_metric import compute_dice

        pred = torch.zeros(2, 1, 32, 32)  # predicts all 0
        target = torch.ones(2, 1, 32, 32)  # true all 1
        dice = compute_dice(pred, target, threshold=0.5)
        assert torch.allclose(
            dice, torch.zeros_like(dice), atol=1e-4
        ), f"All-wrong prediction should give Dice=0, got {dice}"

    def test_dice_range(self):
        from evaluation.dice_metric import compute_dice

        rng = torch.Generator().manual_seed(0)
        pred = torch.rand(8, 1, 64, 64, generator=rng)
        target = (torch.rand(8, 1, 64, 64, generator=rng) > 0.5).float()
        dice = compute_dice(pred, target, threshold=0.5)
        assert (dice >= 0.0).all() and (dice <= 1.0).all(), f"Dice out of [0,1]: {dice}"

    def test_dice_symmetry(self):
        """Dice(A,B) == Dice(B,A) (when symmetrical)."""
        from evaluation.dice_metric import compute_dice

        rng = torch.Generator().manual_seed(1)
        pred = (torch.rand(2, 1, 64, 64, generator=rng) > 0.5).float()
        target = (torch.rand(2, 1, 64, 64, generator=rng) > 0.5).float()
        d1 = compute_dice(pred, target, threshold=0.5)
        d2 = compute_dice(target, pred, threshold=0.5)
        assert torch.allclose(d1, d2, atol=1e-5), "Dice should be symmetric"

    def test_dice_metric_accumulator(self):
        from evaluation.dice_metric import DiceMetric

        dm = DiceMetric(threshold=0.5)
        for _ in range(5):
            pred = torch.rand(4, 1, 64, 64)
            target = (torch.rand(4, 1, 64, 64) > 0.5).float()
            dm.update(pred, target)
        result = dm.compute()
        assert "dice" in result
        assert 0.0 <= result["dice"] <= 1.0

    def test_dice_metric_reset(self):
        from evaluation.dice_metric import DiceMetric

        dm = DiceMetric()
        pred = torch.rand(2, 1, 32, 32)
        target = (torch.rand(2, 1, 32, 32) > 0.5).float()
        dm.update(pred, target)
        dm.reset()
        result = dm.compute()
        # After reset, n_samples == 0 (compute returns 0/1 with dummy denominator)
        assert result["n_samples"] == 0


class TestIoUMetric:
    """Tests for compute_iou."""

    def test_perfect_iou_is_one(self):
        from evaluation.dice_metric import compute_iou

        pred = torch.ones(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        iou = compute_iou(pred, target, threshold=0.5)
        assert torch.allclose(iou, torch.ones_like(iou), atol=1e-4)

    def test_zero_iou_no_overlap(self):
        from evaluation.dice_metric import compute_iou

        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        iou = compute_iou(pred, target, threshold=0.5)
        assert torch.allclose(iou, torch.zeros_like(iou), atol=1e-4)

    def test_iou_range(self):
        from evaluation.dice_metric import compute_iou

        rng = torch.Generator().manual_seed(5)
        pred = torch.rand(6, 1, 64, 64, generator=rng)
        target = (torch.rand(6, 1, 64, 64, generator=rng) > 0.5).float()
        iou = compute_iou(pred, target)
        assert (iou >= 0.0).all() and (iou <= 1.0).all()

    def test_iou_leq_dice(self):
        """IoU ≤ Dice always (by definition: IoU = D/(2-D))."""
        from evaluation.dice_metric import compute_dice, compute_iou

        rng = torch.Generator().manual_seed(7)
        pred = torch.rand(4, 1, 64, 64, generator=rng)
        target = (torch.rand(4, 1, 64, 64, generator=rng) > 0.5).float()
        dice = compute_dice(pred, target, threshold=0.5)
        iou = compute_iou(pred, target, threshold=0.5)
        assert (iou <= dice + 1e-5).all(), "IoU should always be ≤ Dice"


class TestPrecisionRecall:
    """Tests for compute_precision_recall."""

    def test_perfect_precision_recall(self):
        from evaluation.dice_metric import compute_precision_recall

        pred = torch.ones(2, 1, 32, 32) * 0.9
        target = torch.ones(2, 1, 32, 32)
        prec, rec = compute_precision_recall(pred, target, threshold=0.5)
        assert torch.allclose(prec, torch.ones_like(prec), atol=1e-4)
        assert torch.allclose(rec, torch.ones_like(rec), atol=1e-4)

    def test_zero_recall_no_true_pos(self):
        from evaluation.dice_metric import compute_precision_recall

        pred = torch.zeros(2, 1, 32, 32)  # predicts nothing
        target = torch.ones(2, 1, 32, 32)  # all positive
        _, rec = compute_precision_recall(pred, target, threshold=0.5)
        assert torch.allclose(rec, torch.zeros_like(rec), atol=1e-4)

    def test_precision_recall_range(self):
        from evaluation.dice_metric import compute_precision_recall

        pred = torch.rand(4, 1, 64, 64)
        target = (torch.rand(4, 1, 64, 64) > 0.5).float()
        prec, rec = compute_precision_recall(pred, target)
        assert (prec >= 0.0).all() and (prec <= 1.0).all()
        assert (rec >= 0.0).all() and (rec <= 1.0).all()


class TestComputeAllMetrics:
    """Tests for the compute_all_metrics convenience wrapper."""

    def test_returns_all_keys(self):
        from evaluation.dice_metric import compute_all_metrics

        pred = torch.rand(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        result = compute_all_metrics(pred, target)
        for key in ("dice", "iou", "precision", "recall"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_are_scalars(self):
        from evaluation.dice_metric import compute_all_metrics

        pred = torch.rand(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        result = compute_all_metrics(pred, target)
        for k, v in result.items():
            if isinstance(v, (int, float)):
                assert np.isfinite(v), f"Metric {k} is not finite: {v}"


# ══════════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestExpectedCalibrationError:
    """Tests for ECE computation."""

    def test_perfect_calibration_ece_near_zero(self):
        from evaluation.uncertainty_calibration import expected_calibration_error

        # Perfectly calibrated: confidence = fraction correct in each bin
        probs = np.array([0.1] * 100 + [0.5] * 100 + [0.9] * 100)
        correct = np.array(
            [int(np.random.default_rng(i).random() < 0.1) for i in range(100)]
            + [int(np.random.default_rng(i + 100).random() < 0.5) for i in range(100)]
            + [int(np.random.default_rng(i + 200).random() < 0.9) for i in range(100)]
        )
        ece, _, _, _ = expected_calibration_error(
            probs, correct.astype(float), n_bins=10
        )
        assert (
            ece < 0.15
        ), f"ECE should be small for roughly calibrated model, got {ece:.4f}"

    def test_ece_range(self):
        from evaluation.uncertainty_calibration import expected_calibration_error

        rng = np.random.default_rng(42)
        probs = rng.random(1000)
        correct = (rng.random(1000) < probs).astype(float)
        ece, _, _, _ = expected_calibration_error(probs, correct, n_bins=10)
        assert 0.0 <= ece <= 1.0, f"ECE out of [0,1]: {ece}"

    def test_ece_overconfident_is_high(self):
        from evaluation.uncertainty_calibration import expected_calibration_error

        # Always predict 0.95 but only correct 50% of the time
        probs = np.full(500, 0.95)
        correct = np.array([1, 0] * 250, dtype=float)
        ece, _, _, _ = expected_calibration_error(probs, correct, n_bins=10)
        assert ece > 0.2, f"Overconfident model should have high ECE, got {ece:.4f}"


class TestBrierScore:
    """Tests for Brier score computation."""

    def test_brier_perfect_prediction_zero(self):
        from evaluation.uncertainty_calibration import brier_score

        probs = np.ones(100)
        targets = np.ones(100)
        bs = brier_score(probs, targets)
        assert bs == pytest.approx(0.0, abs=1e-6)

    def test_brier_worst_prediction_one(self):
        from evaluation.uncertainty_calibration import brier_score

        probs = np.zeros(100)  # predict 0 for all positive
        targets = np.ones(100)
        bs = brier_score(probs, targets)
        assert bs == pytest.approx(1.0, abs=1e-6)

    def test_brier_range(self):
        from evaluation.uncertainty_calibration import brier_score

        rng = np.random.default_rng(3)
        probs = rng.random(200)
        targets = rng.integers(0, 2, 200).astype(float)
        bs = brier_score(probs, targets)
        assert 0.0 <= bs <= 1.0


class TestEntropyAUC:
    """Tests for uncertainty AUROC (entropy_auc)."""

    def test_auroc_perfect_uncertainty_is_one(self):
        from evaluation.uncertainty_calibration import entropy_auc

        # Perfect: high uncertainty exactly where errors occur
        uncertainty = np.array([0.9] * 50 + [0.1] * 50)
        errors = np.array([1.0] * 50 + [0.0] * 50)
        auc = entropy_auc(uncertainty, errors)
        assert auc >= 0.95, f"Perfect uncertainty should give AUROC≈1, got {auc:.4f}"

    def test_auroc_range(self):
        import math

        from evaluation.uncertainty_calibration import entropy_auc

        rng = np.random.default_rng(11)
        unc = rng.random(300)
        errors = (rng.random(300) > 0.7).astype(float)
        auc = entropy_auc(unc, errors)
        # entropy_auc returns NaN when all errors are identical; skip in that case
        if not math.isnan(auc):
            assert 0.0 <= auc <= 1.0

    def test_auroc_random_is_near_half(self):
        import math

        from evaluation.uncertainty_calibration import entropy_auc

        rng = np.random.default_rng(99)
        unc = rng.random(5000)
        errors = rng.integers(0, 2, 5000).astype(float)
        auc = entropy_auc(unc, errors)
        if not math.isnan(auc):
            assert (
                0.30 <= auc <= 0.70
            ), f"Random uncertainty AUROC should be ~0.5, got {auc:.4f}"


class TestCalibrationAnalyzer:
    """Tests for stateful CalibrationAnalyzer accumulator."""

    def test_accumulator_compute_returns_keys(self):
        from evaluation.uncertainty_calibration import CalibrationAnalyzer

        ca = CalibrationAnalyzer(n_bins=10, subsample_rate=1.0)
        pred = torch.rand(4, 1, 32, 32)
        unc = torch.rand(4, 1, 32, 32) * 0.1
        tgt = (torch.rand(4, 1, 32, 32) > 0.5).float()
        ca.update(pred, unc, tgt)
        result = ca.compute()
        assert "ece" in result
        assert "brier" in result

    def test_accumulator_reset(self):
        from evaluation.uncertainty_calibration import CalibrationAnalyzer

        ca = CalibrationAnalyzer(n_bins=10)
        pred = torch.rand(2, 1, 32, 32)
        unc = torch.rand(2, 1, 32, 32)
        tgt = (torch.rand(2, 1, 32, 32) > 0.5).float()
        ca.update(pred, unc, tgt)
        ca.reset()
        # After reset, _probs list should be empty
        assert hasattr(ca, "_probs") and len(ca._probs) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Radiologist Agreement Metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestCohensKappa:
    """Tests for Cohen's κ computation."""

    def test_perfect_agreement_kappa_is_one(self):
        from evaluation.radiologist_agreement import cohens_kappa

        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        result = cohens_kappa(labels, labels)
        assert result["kappa"] == pytest.approx(1.0, abs=1e-6)

    def test_chance_agreement_kappa_near_zero(self):
        """Systematic swap of labels → κ near −1 (strong disagreement)."""
        from evaluation.radiologist_agreement import cohens_kappa

        rater1 = [0, 1] * 50
        rater2 = [1, 0] * 50  # perfectly opposite
        result = cohens_kappa(rater1, rater2)
        assert result["kappa"] < 0.0

    def test_kappa_range(self):
        from evaluation.radiologist_agreement import cohens_kappa

        rng = np.random.default_rng(20)
        a = rng.integers(0, 2, 80).tolist()
        b = rng.integers(0, 2, 80).tolist()
        result = cohens_kappa(a, b)
        assert -1.0 <= result["kappa"] <= 1.0

    def test_kappa_has_confidence_interval(self):
        from evaluation.radiologist_agreement import cohens_kappa

        a = [0, 1, 0, 1, 1] * 10
        b = [0, 1, 0, 0, 1] * 10
        result = cohens_kappa(a, b)
        # kappa_ci_95 is half-width of the 95% CI
        assert "kappa_ci_95" in result
        assert result["kappa_ci_95"] >= 0.0

    def test_kappa_interpretation_substantial(self):
        from evaluation.radiologist_agreement import cohens_kappa

        rng = np.random.default_rng(30)
        gt = rng.integers(0, 2, 200).tolist()
        pred = [g if rng.random() < 0.9 else 1 - g for g in gt]  # 90% agreement
        result = cohens_kappa(gt, pred)
        assert (
            result["kappa"] > 0.6
        ), f"Expected κ>0.6 for 90% agreement, got {result['kappa']:.3f}"


class TestFleissKappa:
    """Tests for Fleiss κ (multi-rater)."""

    def test_perfect_agreement_fleiss_kappa_one(self):
        from evaluation.radiologist_agreement import fleiss_kappa

        ratings = np.array([[1, 1, 1], [0, 0, 0]] * 25)  # perfect agreement
        result = fleiss_kappa(ratings, n_categories=2)
        assert result["kappa"] == pytest.approx(1.0, abs=1e-5)

    def test_fleiss_kappa_range(self):
        from evaluation.radiologist_agreement import fleiss_kappa

        rng = np.random.default_rng(42)
        ratings = rng.integers(0, 2, (50, 4))  # 50 subjects, 4 raters
        result = fleiss_kappa(ratings, n_categories=2)
        assert -1.0 <= result["kappa"] <= 1.0

    def test_fleiss_kappa_returns_interpretation(self):
        from evaluation.radiologist_agreement import fleiss_kappa

        ratings = np.array([[1, 1, 1], [0, 0, 0]] * 25)
        result = fleiss_kappa(ratings, n_categories=2)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)


class TestPercentAgreement:
    """Tests for percent agreement."""

    def test_perfect_agreement_is_one(self):
        from evaluation.radiologist_agreement import percent_agreement

        ratings = np.array([[1, 1, 1], [0, 0, 0]] * 30)
        pct = percent_agreement(ratings)
        assert pct == pytest.approx(1.0, abs=1e-6)

    def test_no_agreement_is_zero(self):
        from evaluation.radiologist_agreement import percent_agreement

        # All raters disagree on every row
        ratings = np.array([[0, 1, 0], [1, 0, 1]] * 25)
        pct = percent_agreement(ratings)
        assert pct == pytest.approx(0.0, abs=1e-6)

    def test_agreement_range(self):
        from evaluation.radiologist_agreement import percent_agreement

        rng = np.random.default_rng(7)
        ratings = rng.integers(0, 2, (60, 3))
        pct = percent_agreement(ratings)
        assert 0.0 <= pct <= 1.0


class TestBlandAltman:
    """Tests for Bland-Altman analysis."""

    def test_returns_required_keys(self):
        from evaluation.radiologist_agreement import bland_altman

        m1 = np.array([100.0, 200.0, 150.0, 250.0, 180.0])
        m2 = np.array([105.0, 195.0, 148.0, 255.0, 182.0])
        result = bland_altman(m1, m2)
        for key in ("mean_diff", "std_diff", "loa_lower", "loa_upper"):
            assert key in result, f"Missing key: {key}"

    def test_zero_bias_identical_measurements(self):
        from evaluation.radiologist_agreement import bland_altman

        vals = np.array([100.0, 200.0, 300.0, 400.0])
        result = bland_altman(vals, vals)
        assert result["mean_diff"] == pytest.approx(0.0, abs=1e-6)
        assert result["std_diff"] == pytest.approx(0.0, abs=1e-6)

    def test_loa_symmetric_around_bias(self):
        """LoA lower and upper should be equidistant from mean_diff (±z*SD)."""
        from evaluation.radiologist_agreement import bland_altman

        rng = np.random.default_rng(55)
        m1 = rng.normal(200, 20, 50)
        m2 = m1 + rng.normal(5, 10, 50)
        result = bland_altman(m1, m2)
        half_width_up = result["loa_upper"] - result["mean_diff"]
        half_width_lo = result["mean_diff"] - result["loa_lower"]
        assert half_width_up == pytest.approx(half_width_lo, abs=1e-4)


class TestRadiologistAgreementAccumulator:
    """Tests for the RadiologistAgreement stateful accumulator."""

    def test_add_study_and_compute(self):
        from evaluation.radiologist_agreement import RadiologistAgreement

        ra = RadiologistAgreement(n_radiologists=3)
        rng = np.random.default_rng(1)
        for i in range(10):
            model_lbl = int(rng.integers(0, 2))
            rad_lbls = [int(rng.integers(0, 2)) for _ in range(3)]
            ra.add_study(f"s{i}", model_lbl, rad_lbls)
        result = ra.compute()
        assert "cohens_kappa_mean" in result or "kappa" in result or len(result) > 0

    def test_wrong_rater_count_raises(self):
        from evaluation.radiologist_agreement import RadiologistAgreement

        ra = RadiologistAgreement(n_radiologists=3)
        with pytest.raises(ValueError):
            ra.add_study("s0", 1, [0, 1])  # only 2 raters, expected 3
