"""
Radiologist agreement analysis for clinical validation.

This module computes the inter-rater agreement statistics reported in
Section 5.3 of the project report:

- **Cohen's κ** — pairwise agreement between model and each radiologist.
- **Fleiss' κ** — multi-rater agreement across all radiologists simultaneously.
- **Percent Agreement** — raw fraction of cases where all raters agree.
- **Bland-Altman analysis** — limits of agreement for continuous volume measurements.
- **RadiologistAgreement** — stateful accumulator for the 150-study clinical study.

Reported values (Table 4 of paper):
  Mean Cohen's κ = 0.83, 91% radiologist agreement.

Run this file for a demo::

    python evaluation/radiologist_agreement.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------


def cohens_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    n_categories: int = 2,
) -> Dict[str, float]:
    """Compute Cohen's κ for two raters on ordinal / categorical data.

    Cohen's kappa accounts for chance agreement::

        κ = (p_o − p_e) / (1 − p_e)

    where p_o is observed agreement and p_e is expected agreement by chance.

    Parameters
    ----------
    rater_a : sequence of int
        Ratings from rater A, length N.
    rater_b : sequence of int
        Ratings from rater B, length N. Same length as rater_a.
    n_categories : int
        Number of categories (2 for binary: malignant/benign). Default 2.

    Returns
    -------
    dict with keys:
        ``kappa``       : Cohen's κ ∈ [-1, 1]. 1 = perfect, 0 = chance.
        ``p_observed``  : Raw observed agreement fraction.
        ``p_expected``  : Expected agreement under independence.
        ``n``           : Number of cases.
        ``interpretation`` : Landis & Koch (1977) label.

    Raises
    ------
    ValueError
        If rater sequences have different lengths or are empty.
    """
    ra = np.asarray(rater_a, dtype=int)
    rb = np.asarray(rater_b, dtype=int)
    if len(ra) != len(rb):
        raise ValueError(f"rater_a and rater_b must have equal length: {len(ra)} vs {len(rb)}")
    if len(ra) == 0:
        raise ValueError("Rater sequences must not be empty")

    N = len(ra)

    # Observed agreement
    p_observed = float((ra == rb).mean())

    # Expected agreement: sum over categories of (freq_a × freq_b)
    categories = list(range(n_categories))
    p_expected = 0.0
    for c in categories:
        freq_a = (ra == c).mean()
        freq_b = (rb == c).mean()
        p_expected += freq_a * freq_b

    # κ
    denom = 1.0 - p_expected
    if abs(denom) < 1e-10:
        kappa = 1.0 if abs(p_observed - 1.0) < 1e-10 else 0.0
    else:
        kappa = (p_observed - p_expected) / denom

    # Confidence interval (Fleiss, 1969 standard error approximation)
    se = np.sqrt(p_observed * (1 - p_observed) / N) if N > 0 else 0.0
    ci_95 = 1.96 * se

    # Landis & Koch interpretation
    interpretation = _landis_koch(kappa)

    return {
        "kappa": float(kappa),
        "p_observed": p_observed,
        "p_expected": p_expected,
        "n": float(N),
        "kappa_ci_95": float(ci_95),
        "interpretation": interpretation,
    }


def _landis_koch(kappa: float) -> str:
    """Landis & Koch (1977) strength-of-agreement labels for κ."""
    if kappa < 0.0:
        return "Poor (< 0)"
    elif kappa < 0.20:
        return "Slight (0.00–0.20)"
    elif kappa < 0.40:
        return "Fair (0.21–0.40)"
    elif kappa < 0.60:
        return "Moderate (0.41–0.60)"
    elif kappa < 0.80:
        return "Substantial (0.61–0.80)"
    else:
        return "Almost perfect (0.81–1.00)"


# ---------------------------------------------------------------------------
# Fleiss' κ
# ---------------------------------------------------------------------------


def fleiss_kappa(
    ratings_matrix: np.ndarray,
    n_categories: int = 2,
) -> Dict[str, float]:
    """Compute Fleiss' κ for k ≥ 2 raters and N subjects.

    Parameters
    ----------
    ratings_matrix : np.ndarray
        Shape (N, k). Each row is one subject, each column is one rater.
        Values are integer category labels in range [0, n_categories-1].
    n_categories : int
        Number of rating categories. Default 2 (binary).

    Returns
    -------
    dict with keys:
        ``kappa`` : Fleiss' κ ∈ [-1, 1].
        ``p_o``   : Observed agreement.
        ``p_e``   : Expected agreement.
        ``n``     : Number of subjects.
        ``k``     : Number of raters.
        ``interpretation`` : Landis & Koch label.
    """
    ratings = np.asarray(ratings_matrix, dtype=int)
    N, k = ratings.shape  # subjects × raters

    # Build count matrix: n_ij = # raters who assigned category j to subject i
    counts = np.zeros((N, n_categories), dtype=float)
    for j in range(n_categories):
        counts[:, j] = (ratings == j).sum(axis=1)

    # Per-subject agreement: P_i = (1 / k(k-1)) * Σ_j n_ij(n_ij - 1)
    P_i = (counts * (counts - 1)).sum(axis=1) / (k * (k - 1))
    p_o = P_i.mean()

    # Expected agreement: p_j = proportion of all ratings in category j
    p_j = counts.sum(axis=0) / (N * k)
    p_e = (p_j ** 2).sum()

    denom = 1.0 - p_e
    if abs(denom) < 1e-10:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-10 else 0.0
    else:
        kappa = (p_o - p_e) / denom

    return {
        "kappa": float(kappa),
        "p_o": float(p_o),
        "p_e": float(p_e),
        "n": float(N),
        "k": float(k),
        "interpretation": _landis_koch(kappa),
    }


# ---------------------------------------------------------------------------
# Percent agreement
# ---------------------------------------------------------------------------


def percent_agreement(
    ratings_matrix: np.ndarray,
) -> float:
    """Compute fraction of cases where all raters agree.

    Parameters
    ----------
    ratings_matrix : np.ndarray
        Shape (N, k). Rater ratings per case.

    Returns
    -------
    float
        Fraction of rows where all k raters gave the same rating ∈ [0, 1].
    """
    ratings = np.asarray(ratings_matrix, dtype=int)
    # All raters agree ↔ max == min along rater axis
    agree = (ratings.max(axis=1) == ratings.min(axis=1))
    return float(agree.mean())


# ---------------------------------------------------------------------------
# Bland-Altman analysis
# ---------------------------------------------------------------------------


def bland_altman(
    method_a: np.ndarray,
    method_b: np.ndarray,
    units: str = "mm³",
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Bland-Altman limits of agreement for nodule volume measurements.

    Computes the mean difference (bias) and ±1.96 SD limits of agreement
    between two measurement methods (e.g., model vs. radiologist volume).

    Parameters
    ----------
    method_a : np.ndarray
        Measurements from method A (e.g., model), shape (N,).
    method_b : np.ndarray
        Measurements from method B (e.g., radiologist), shape (N,).
    units : str
        Measurement units for reporting only. Default ``"mm³"``.
    confidence : float
        Confidence interval multiplier. 0.95 → 1.96 × SD. Default 0.95.

    Returns
    -------
    dict with keys:
        ``mean_diff``      : Mean of (A − B) = bias.
        ``std_diff``       : SD of (A − B).
        ``loa_upper``      : Upper limit of agreement.
        ``loa_lower``      : Lower limit of agreement.
        ``mean_of_means``  : Mean of (A + B) / 2.
        ``n``              : Number of paired measurements.
        ``units``          : Units string.
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"method_a and method_b must have equal length: {len(a)} vs {len(b)}")

    diff = a - b
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))

    # z for given confidence level (approximation for large N)
    import scipy.stats
    z = float(scipy.stats.norm.ppf((1 + confidence) / 2))

    loa_upper = mean_diff + z * std_diff
    loa_lower = mean_diff - z * std_diff

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "mean_of_means": float(((a + b) / 2).mean()),
        "n": float(len(a)),
        "units": units,
    }


def plot_bland_altman(
    method_a: np.ndarray,
    method_b: np.ndarray,
    ba_stats: Dict[str, float],
    title: str = "Bland-Altman Plot",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot a Bland-Altman agreement plot.

    Parameters
    ----------
    method_a : np.ndarray
        Measurements from method A.
    method_b : np.ndarray
        Measurements from method B.
    ba_stats : dict
        Output of ``bland_altman()``.
    title : str
        Plot title. Default "Bland-Altman Plot".
    save_path : str or None
        Save PNG to this path if provided.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _MPL_AVAILABLE:
        logger.warning("plot_bland_altman: matplotlib not available")
        return None

    a, b = np.asarray(method_a), np.asarray(method_b)
    means = (a + b) / 2
    diffs = a - b
    units = ba_stats.get("units", "")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, diffs, alpha=0.6, s=30, color="#4C72B0", label="Observations")
    ax.axhline(ba_stats["mean_diff"], color="#C44E52", linewidth=2, label=f"Bias = {ba_stats['mean_diff']:.2f} {units}")
    ax.axhline(ba_stats["loa_upper"], color="#DD8452", linewidth=1.5, linestyle="--",
               label=f"LoA upper = {ba_stats['loa_upper']:.2f} {units}")
    ax.axhline(ba_stats["loa_lower"], color="#DD8452", linewidth=1.5, linestyle="--",
               label=f"LoA lower = {ba_stats['loa_lower']:.2f} {units}")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel(f"Mean of model + radiologist ({units})", fontsize=11)
    ax.set_ylabel(f"Difference (model − radiologist) ({units})", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Bland-Altman plot saved → %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Stateful accumulator for the 150-study clinical study
# ---------------------------------------------------------------------------


class RadiologistAgreement:
    """Accumulate and analyse radiologist vs. model agreement over clinical studies.

    Designed for the 150-study clinical validation protocol. After accumulating
    per-study observations with ``add_study()``, call ``compute()`` to get the
    full agreement analysis including Cohen's κ, Fleiss' κ, and Bland-Altman.

    Parameters
    ----------
    n_radiologists : int
        Number of radiologists in the study. Default 3.
    volume_units : str
        Units for nodule volume measurements. Default ``"mm³"``.
    """

    def __init__(self, n_radiologists: int = 3, volume_units: str = "mm³") -> None:
        self.n_radiologists = n_radiologists
        self.volume_units = volume_units
        # Binary nodule classification lists
        self._model_labels: List[int] = []
        # rater_labels[i] = list of labels from radiologist i
        self._rater_labels: List[List[int]] = [[] for _ in range(n_radiologists)]
        # Volume measurements (model vs. mean radiologist)
        self._model_volumes: List[float] = []
        self._rad_volumes: List[List[float]] = [[] for _ in range(n_radiologists)]
        self._study_ids: List[str] = []

    def add_study(
        self,
        study_id: str,
        model_label: int,
        radiologist_labels: Sequence[int],
        model_volume_mm3: Optional[float] = None,
        radiologist_volumes_mm3: Optional[Sequence[float]] = None,
    ) -> None:
        """Record results for one CT study.

        Parameters
        ----------
        study_id : str
            Unique study identifier.
        model_label : int
            Model's binary classification (0 = no nodule, 1 = nodule).
        radiologist_labels : sequence of int
            One label per radiologist. Length must equal ``n_radiologists``.
        model_volume_mm3 : float or None
            Model's nodule volume estimate. None to skip Bland-Altman.
        radiologist_volumes_mm3 : sequence of float or None
            One volume estimate per radiologist.

        Raises
        ------
        ValueError
            If ``radiologist_labels`` length doesn't match ``n_radiologists``.
        """
        if len(radiologist_labels) != self.n_radiologists:
            raise ValueError(
                f"Expected {self.n_radiologists} radiologist labels, "
                f"got {len(radiologist_labels)}"
            )
        self._study_ids.append(study_id)
        self._model_labels.append(int(model_label))
        for i, lbl in enumerate(radiologist_labels):
            self._rater_labels[i].append(int(lbl))

        if model_volume_mm3 is not None and radiologist_volumes_mm3 is not None:
            self._model_volumes.append(float(model_volume_mm3))
            for i, vol in enumerate(radiologist_volumes_mm3):
                self._rad_volumes[i].append(float(vol))

    def compute(
        self,
        save_ba_plot: Optional[str] = None,
    ) -> Dict[str, object]:
        """Compute full agreement analysis.

        Parameters
        ----------
        save_ba_plot : str or None
            If provided, save Bland-Altman plot to this path.

        Returns
        -------
        dict with keys:
            ``n_studies``, ``percent_agreement``,
            ``cohen_kappa_per_rater`` (list), ``cohen_kappa_mean``,
            ``cohen_kappa_std``, ``fleiss_kappa``,
            ``bland_altman`` (dict or None).
        """
        N = len(self._study_ids)
        if N == 0:
            return {"n_studies": 0}

        model_arr = np.array(self._model_labels, dtype=int)

        # Build full ratings matrix: rows = studies, columns = [model, rad1, rad2, ...]
        all_rater_cols = [model_arr]
        for rater in self._rater_labels:
            all_rater_cols.append(np.array(rater, dtype=int))
        ratings_matrix = np.column_stack(all_rater_cols)  # (N, n_rads+1)

        pct_agree = percent_agreement(ratings_matrix)

        # Cohen's κ: model vs. each radiologist
        kappas: List[float] = []
        kappa_details: List[Dict] = []
        for i, rater in enumerate(self._rater_labels):
            result = cohens_kappa(model_arr.tolist(), rater, n_categories=2)
            kappas.append(result["kappa"])
            kappa_details.append(result)

        kappa_mean = float(np.mean(kappas))
        kappa_std = float(np.std(kappas, ddof=1)) if len(kappas) > 1 else 0.0

        # Fleiss' κ over all raters
        fleiss = fleiss_kappa(ratings_matrix, n_categories=2)

        output: Dict[str, object] = {
            "n_studies": N,
            "percent_agreement": pct_agree,
            "cohen_kappa_per_rater": kappas,
            "cohen_kappa_details": kappa_details,
            "cohen_kappa_mean": kappa_mean,
            "cohen_kappa_std": kappa_std,
            "fleiss_kappa": fleiss,
        }

        # Bland-Altman (if volumes provided)
        ba_stats = None
        if self._model_volumes:
            rad_mean_vol = np.mean(np.column_stack(self._rad_volumes), axis=1)
            model_vol = np.asarray(self._model_volumes)
            ba_stats = bland_altman(model_vol, rad_mean_vol, units=self.volume_units)
            output["bland_altman"] = ba_stats

            if save_ba_plot:
                plot_bland_altman(
                    model_vol, rad_mean_vol, ba_stats,
                    title=f"Bland-Altman: Model vs Radiologists (n={N})",
                    save_path=save_ba_plot,
                )
        else:
            output["bland_altman"] = None

        # Log summary
        logger.info(
            "RadiologistAgreement: N=%d | %%agree=%.1f%% | "
            "Cohen κ=%.3f±%.3f | Fleiss κ=%.3f | interp=%s",
            N, pct_agree * 100,
            kappa_mean, kappa_std,
            fleiss["kappa"], fleiss["interpretation"],
        )

        return output

    def summary_string(self) -> str:
        """Return a human-readable summary string (without computing full analysis)."""
        return (
            f"RadiologistAgreement: {len(self._study_ids)} studies | "
            f"{self.n_radiologists} radiologists"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import scipy.stats

    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("RadiologistAgreement — demo (150-study simulation)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    N_STUDIES = 150
    N_RADS = 3

    # Simulate ground truth labels with 91% model agreement
    gt_labels = rng.binomial(1, 0.5, N_STUDIES)
    model_labels = gt_labels.copy()
    flip_mask = rng.random(N_STUDIES) < 0.09   # 9% error rate → 91% agreement
    model_labels[flip_mask] = 1 - model_labels[flip_mask]

    # Radiologists agree with GT ~93% of the time each
    rad_labels = []
    for _ in range(N_RADS):
        rl = gt_labels.copy()
        flip = rng.random(N_STUDIES) < 0.07
        rl[flip] = 1 - rl[flip]
        rad_labels.append(rl)

    # Volume measurements: model slightly over-estimates vs radiologist mean
    model_volumes = rng.normal(450, 120, N_STUDIES).clip(50, 2000)
    rad_volumes_list = [rng.normal(440, 110, N_STUDIES).clip(50, 2000) for _ in range(N_RADS)]

    # Build RadiologistAgreement
    ra = RadiologistAgreement(n_radiologists=N_RADS)
    for i in range(N_STUDIES):
        ra.add_study(
            study_id=f"LUNA_{i:04d}",
            model_label=int(model_labels[i]),
            radiologist_labels=[int(rad_labels[r][i]) for r in range(N_RADS)],
            model_volume_mm3=float(model_volumes[i]),
            radiologist_volumes_mm3=[float(rad_volumes_list[r][i]) for r in range(N_RADS)],
        )

    results = ra.compute(save_ba_plot="bland_altman_demo.png")

    print(f"\nn_studies:          {results['n_studies']}")
    print(f"percent_agreement:  {results['percent_agreement']:.3f}  ({results['percent_agreement']*100:.1f}%)")
    print(f"Cohen κ (mean):     {results['cohen_kappa_mean']:.4f} ± {results['cohen_kappa_std']:.4f}")
    print(f"Cohen κ per rater:  {[f'{k:.4f}' for k in results['cohen_kappa_per_rater']]}")
    fk = results['fleiss_kappa']
    print(f"Fleiss κ:           {fk['kappa']:.4f}  ({fk['interpretation']})")
    ba = results['bland_altman']
    print(f"Bland-Altman bias:  {ba['mean_diff']:.2f} mm³  LoA=[{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]")

    # Assertions matching paper targets
    assert results['percent_agreement'] > 0.85, "Agreement should be > 85%"
    assert results['cohen_kappa_mean'] > 0.70, "κ should be > 0.70"
    print("\nAll assertions passed. ✓")
    print("Saved bland_altman_demo.png")
