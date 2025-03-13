"""
Learning-rate schedulers for SAM2 lung nodule segmentation training.

Provides a custom ``WarmupCosineScheduler`` and a ``build_scheduler``
factory that reads the ``scheduler`` section of ``training/config.yaml``
to return the appropriate ``torch.optim.lr_scheduler``.

Supported schedulers (``scheduler.name`` in config.yaml):

- ``cosine_with_warmup`` — linear warmup → cosine annealing (default).
- ``step``               — ``StepLR``.
- ``plateau``            — ``ReduceLROnPlateau``.
- ``poly``               — polynomial decay.

Run this file directly for a visual LR plot sanity check::

    python training/lr_scheduler.py
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Union

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WarmupCosineScheduler
# ---------------------------------------------------------------------------


class WarmupCosineScheduler(sched.LRScheduler):
    """Cosine annealing with linear warm-up.

    During the first ``warmup_epochs``, the LR grows linearly from
    ``warmup_start_lr`` to the optimizer's base LR (``base_lrs``).
    After warm-up, it anneals following the standard cosine schedule
    down to ``eta_min``, completing one full cosine cycle over ``T_max``
    epochs (counting from the start of warm-up).

    The schedule is identical to that used in the published experiments
    (Section 3.2 of the project report).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    warmup_epochs : int
        Number of epochs for the linear warm-up phase. Default 5.
    T_max : int
        Total number of epochs (including warm-up) for one cosine cycle.
        After ``T_max`` epochs the LR stays at ``eta_min``. Default 50.
    warmup_start_lr : float
        LR at epoch 0 (start of warm-up). Default 1e-7.
    eta_min : float
        Minimum LR at the end of the cosine phase. Default 1e-7.
    last_epoch : int
        The index of the last epoch. -1 (default) initialises the schedule.

    Notes
    -----
    ``epoch`` here means the integer step passed to ``scheduler.step()``.
    Call ``scheduler.step()`` **once per epoch**, not once per batch.

    Examples
    --------
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    >>> scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, T_max=50)
    >>> for epoch in range(50):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 5,
        T_max: int = 50,
        warmup_start_lr: float = 1e-7,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        """Compute the current LR for each parameter group.

        Returns
        -------
        list of float
            One LR value per parameter group in the optimizer.
        """
        epoch = self.last_epoch  # 0-indexed

        if epoch < self.warmup_epochs:
            # Linear warm-up: warmup_start_lr → base_lr
            warmup_frac = epoch / max(self.warmup_epochs, 1)
            return [
                self.warmup_start_lr + warmup_frac * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing after warm-up
            # Progress ∈ [0, 1] through the cosine phase
            cosine_epoch = epoch - self.warmup_epochs
            cosine_total = max(self.T_max - self.warmup_epochs, 1)
            progress = cosine_epoch / cosine_total
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_mult
                for base_lr in self.base_lrs
            ]


# ---------------------------------------------------------------------------
# Polynomial LR Scheduler (PyTorch 1.x compatible)
# ---------------------------------------------------------------------------


class PolynomialLR(sched.LRScheduler):
    """Polynomial learning rate decay.

    LR decays from ``base_lr`` to ``eta_min`` following a power law::

        lr_t = eta_min + (base_lr - eta_min) * (1 - t / total_iters) ** power

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    total_iters : int
        Total number of decay steps (epochs). Default 50.
    power : float
        Polynomial exponent. 1.0 → linear decay; 2.0 → quadratic. Default 0.9.
    eta_min : float
        Minimum LR. Default 0.0.
    last_epoch : int
        Start epoch index (-1 = fresh start). Default -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_iters: int = 50,
        power: float = 0.9,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = total_iters
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        t = min(self.last_epoch, self.total_iters)
        factor = (1.0 - t / self.total_iters) ** self.power
        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
) -> Union[sched.LRScheduler, sched.ReduceLROnPlateau]:
    """Build the appropriate LR scheduler from ``config["scheduler"]``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose LR will be scheduled.
    config : dict
        Full training config dict (loaded from ``config.yaml``).
        Reads the ``scheduler`` sub-dict.

    Returns
    -------
    ``LRScheduler`` or ``ReduceLROnPlateau``
        The constructed scheduler.

    Raises
    ------
    ValueError
        If ``scheduler.name`` is not one of the supported values.

    Examples
    --------
    >>> import yaml
    >>> cfg = yaml.safe_load(open("training/config.yaml"))
    >>> scheduler = build_scheduler(optimizer, cfg)
    >>> scheduler.step()  # call once per epoch
    """
    sched_cfg: Dict[str, Any] = config.get("scheduler", {})
    name: str = sched_cfg.get("name", "cosine_with_warmup")
    total_epochs: int = config.get("training", {}).get("epochs", 50)

    logger.info("build_scheduler: name=%s | total_epochs=%d", name, total_epochs)

    if name == "cosine_with_warmup":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=int(sched_cfg.get("warmup_epochs", 5)),
            T_max=int(sched_cfg.get("T_max", total_epochs)),
            warmup_start_lr=float(sched_cfg.get("warmup_start_lr", 1e-7)),
            eta_min=float(sched_cfg.get("eta_min", 1e-7)),
        )

    elif name == "step":
        scheduler = sched.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 15)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )

    elif name == "plateau":
        scheduler = sched.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("plateau_mode", "min"),
            patience=int(sched_cfg.get("plateau_patience", 5)),
            factor=float(sched_cfg.get("plateau_factor", 0.5)),
            min_lr=float(sched_cfg.get("eta_min", 1e-7)),
        )

    elif name == "poly":
        scheduler = PolynomialLR(
            optimizer,
            total_iters=int(sched_cfg.get("poly_total_iters", total_epochs)),
            power=float(sched_cfg.get("poly_power", 0.9)),
            eta_min=float(sched_cfg.get("eta_min", 0.0)),
        )

    else:
        raise ValueError(
            f"Unknown scheduler name: {name!r}. "
            "Choose from: cosine_with_warmup, step, plateau, poly"
        )

    return scheduler


def get_lr(optimizer: optim.Optimizer) -> float:
    """Return the current LR of the first optimizer parameter group.

    Convenience helper for logging.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer.

    Returns
    -------
    float
        Current LR.
    """
    return optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=" * 65)
    print("LR Scheduler — sanity check (plots saved to lr_schedules.png)")
    print("=" * 65)

    BASE_LR = 1e-4
    EPOCHS = 50

    def _make_opt(lr: float = BASE_LR) -> optim.Optimizer:
        model = torch.nn.Linear(10, 1)
        return optim.AdamW(model.parameters(), lr=lr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("LR Schedule Sanity Check", fontsize=14, fontweight="bold")

    configs_to_test = [
        {
            "name": "cosine_with_warmup",
            "scheduler": {
                "name": "cosine_with_warmup",
                "warmup_epochs": 5,
                "T_max": 50,
                "eta_min": 1e-7,
            },
            "training": {"epochs": EPOCHS},
        },
        {
            "name": "step",
            "scheduler": {"name": "step", "step_size": 15, "gamma": 0.5},
            "training": {"epochs": EPOCHS},
        },
        {
            "name": "plateau (simulated)",
            "scheduler": {
                "name": "plateau",
                "plateau_mode": "min",
                "plateau_patience": 5,
                "plateau_factor": 0.5,
            },
            "training": {"epochs": EPOCHS},
        },
        {
            "name": "poly",
            "scheduler": {
                "name": "poly",
                "poly_total_iters": EPOCHS,
                "poly_power": 0.9,
                "eta_min": 1e-7,
            },
            "training": {"epochs": EPOCHS},
        },
    ]

    # Assertions
    for ax, cfg in zip(axes.flatten(), configs_to_test):
        opt = _make_opt(BASE_LR)
        is_plateau = cfg["scheduler"]["name"] == "plateau"
        scheduler = build_scheduler(opt, cfg)
        lrs: List[float] = []
        for e in range(EPOCHS):
            lrs.append(get_lr(opt))
            if is_plateau:
                # Simulate a flat validation loss to trigger LR drops
                loss_val = 0.5 if e < 10 else (0.4 if e < 25 else 0.3)
                scheduler.step(loss_val)
            else:
                scheduler.step()

        ax.plot(range(EPOCHS), lrs, linewidth=2)
        ax.set_title(cfg["name"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        print(
            f"  [{cfg['name']}] LR[0]={lrs[0]:.2e}  LR[5]={lrs[5]:.2e}  LR[-1]={lrs[-1]:.2e}"
        )

    plt.tight_layout()
    plt.savefig("lr_schedules.png", dpi=120, bbox_inches="tight")
    print("\nSaved lr_schedules.png")

    # WarmupCosine: explicit assertions
    opt2 = _make_opt(BASE_LR)
    wcs = WarmupCosineScheduler(
        opt2, warmup_epochs=5, T_max=50, warmup_start_lr=1e-7, eta_min=1e-7
    )
    wcs_lrs = []
    for _ in range(50):
        wcs_lrs.append(get_lr(opt2))
        wcs.step()

    peak_lr = max(wcs_lrs)
    assert abs(peak_lr - BASE_LR) < 1e-8, f"Peak LR should be ≈ base_lr, got {peak_lr}"
    assert wcs_lrs[0] < wcs_lrs[4] < peak_lr, "LR should rise during warmup"
    assert wcs_lrs[-1] <= peak_lr * 0.1, "LR should drop substantially after cosine"
    print("\nWarmupCosineScheduler assertions: ✓")
    print("All scheduler checks passed. ✓")
