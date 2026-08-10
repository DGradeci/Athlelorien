"""
Shared utilities for the Levy-type transport paper.

Covers:
  - Paths / directory constants
  - Analysis parameters (shared across all notebooks)
  - Publication-quality plotting configuration
  - Reusable panel builders (CCDF, MSD, hazard, state matrix)
  - Cache helpers (save/load parquet artefacts)

Usage
-----
    from analysis.levy_paper.util.paper_utils import (
        configure_paper_plotting,
        THETA_DEG, STATE_COLORS, DATA_DIR, FIGURES_DIR,
        plot_ccdf, plot_msd, plot_hazard,
        save_cache, load_cache,
    )
    configure_paper_plotting()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent          # .../analysis/levy_paper/util
_PAPER_ROOT = _HERE.parent                       # .../analysis/levy_paper
_REPO_ROOT  = _PAPER_ROOT.parent.parent          # .../athlelorian

DATA_DIR    = _PAPER_ROOT / "data"
FIGURES_DIR = _PAPER_ROOT / "figures"
PRIMARY_CACHE_NAME = "all_team_2020_2021_sticky_active"
PRIMARY_CACHE_DIR = DATA_DIR / "processed" / PRIMARY_CACHE_NAME

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Make repo root importable so `from src.utils import ...` works in notebooks
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Analysis parameters  (single source of truth across all notebooks)
# ---------------------------------------------------------------------------

# Run segmentation
THETA_DEG       = 30      # turning-angle threshold (degrees)
DT              = 1.0     # sampling interval (seconds)
MIN_RUN_FRAMES  = 3       # shortest run to keep

# Active-player geometry
ACTIVE_DEPTH_M  = 6.0     # distance from goal line to classify as active
ACTIVATE_S      = 70      # hysteresis: frames in-zone before "active"
BENCH_OFF_S     = 110     # frames out-of-zone before "bench"
ACTIVE_PLAYER_METHOD = "sticky_hierarchical_active_xi"
MAX_ACTIVE_PLAYERS = 11

# Hazard model
A0              = 5.0     # age-offset in inverse-age baseline
MAX_T           = 120     # maximum run age considered (seconds)

# Stochastic model
N_SIM           = 20_000  # simulation draws for CCDF comparison

# Collective-order state boundaries (polarisation p ∈ [0, 1])
STATE_BOUNDS    = [0.0, 1/3, 2/3, 1.0]      # tercile cut-points
STATE_LABELS    = ["low", "mid", "high"]

# Robustness grid
THETA_GRID      = [20, 30, 40]               # turning-angle sensitivity sweep


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour palette (colour-blind friendly)
STATE_COLORS: dict[str, str] = {
    "low":  "#4393c3",   # blue
    "mid":  "#f4a582",   # orange
    "high": "#d6604d",   # red
}

TEAM_COLORS: dict[str, str] = {
    "A": "#1b7837",   # green  (focal team)
    "B": "#762a83",   # purple (opponent)
}

_LINESTYLES = {"low": "--", "mid": ":", "high": "-"}


def configure_paper_plotting(base: int = 11) -> None:
    """Apply publication-quality Matplotlib settings."""
    plt.style.use("default")
    try:
        import seaborn as sns
        sns.reset_orig()
    except Exception:
        pass

    mpl.rcParams.update(
        {
            "text.usetex":        False,
            "mathtext.fontset":   "cm",
            "font.family":        "DejaVu Serif",
            "font.serif": ["DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype":       42,
            "ps.fonttype":        42,
            # Sizes
            "font.size":              base,
            "axes.titlesize":         base * 1.15,
            "axes.labelsize":         base,
            "xtick.labelsize":        base * 0.9,
            "ytick.labelsize":        base * 0.9,
            "legend.fontsize":        base * 0.85,
            "legend.title_fontsize":  base * 0.9,
            "figure.titlesize":       base * 1.3,
            # Colours
            "text.color":        "black",
            "axes.labelcolor":   "black",
            "axes.titlecolor":   "black",
            "axes.edgecolor":    "black",
            "xtick.color":       "black",
            "ytick.color":       "black",
            # Backgrounds
            "figure.facecolor":  "white",
            "axes.facecolor":    "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi":       300,
        }
    )


def draw_panel_letter(
    ax: plt.Axes,
    letter: str,
    *,
    x: float = -0.12,
    y: float = 1.04,
    fontsize: float | None = None,
    weight: str = "normal",
    suffix: str = ".",
) -> None:
    """Draw a standard letter-only panel label in axes coordinates."""
    ax.text(
        x,
        y,
        f"{letter}{suffix}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight=weight,
        color="black",
        clip_on=False,
    )


def draw_panel_letters(
    axes: Sequence[plt.Axes],
    *,
    letters: Sequence[str] | None = None,
    x: float = -0.12,
    y: float = 1.04,
    fontsize: float | None = None,
    weight: str = "normal",
    suffix: str = ".",
) -> None:
    """Apply standard letter-only panel labels to a sequence of axes."""
    if letters is None:
        letters = [chr(ord("A") + i) for i in range(len(axes))]
    for ax, letter in zip(axes, letters):
        draw_panel_letter(ax, letter, x=x, y=y, fontsize=fontsize, weight=weight, suffix=suffix)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def plot_ccdf(
    ax: plt.Axes,
    values: np.ndarray | pd.Series,
    *,
    label: str = "",
    color: str = "black",
    lw: float = 1.5,
    ls: str = "-",
    reference_exp: bool = False,
) -> None:
    """
    Plot a complementary CDF (CCDF) on *ax* as a log-log line.

    Parameters
    ----------
    ax           : Axes to draw on.
    values       : 1-D array of positive run durations or lengths.
    label        : Legend label.
    color        : Line colour.
    lw           : Line width.
    ls           : Line style.
    reference_exp: If True, overlay an exponential reference line.
    """
    x = np.sort(np.asarray(values, dtype=float))
    x = x[x > 0]
    n = len(x)
    y = 1.0 - np.arange(1, n + 1) / n  # survival fraction

    ax.plot(x, y, color=color, lw=lw, ls=ls, label=label)

    if reference_exp:
        lam = 1.0 / np.mean(x)
        x_ref = np.linspace(x[0], x[-1], 300)
        ax.plot(x_ref, np.exp(-lam * x_ref), color="grey", lw=1.0,
                ls=":", label="Exp. ref.", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.set_xlabel("Value")
    ax.set_ylabel(r"$P(\geq x)$")


def plot_ccdf_by_state(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    state_col: str = "order_state",
    *,
    labels: Sequence[str] | None = None,
) -> None:
    """
    Overlay CCDFs for each collective-order state on *ax*.

    Parameters
    ----------
    df         : DataFrame with *value_col* and *state_col*.
    value_col  : Column of positive run durations or lengths.
    state_col  : Column identifying the order state ("low"/"mid"/"high").
    labels     : Override legend labels.
    """
    states = labels if labels is not None else STATE_LABELS
    for state in states:
        sub = df.loc[df[state_col] == state, value_col].dropna()
        if len(sub) == 0:
            continue
        plot_ccdf(
            ax, sub,
            label=state.capitalize(),
            color=STATE_COLORS[state],
            ls=_LINESTYLES[state],
        )
    ax.legend(title="Order state")


def plot_msd(
    ax: plt.Axes,
    tau: np.ndarray,
    msd: np.ndarray,
    *,
    label: str = "",
    color: str = "black",
    lw: float = 1.5,
    fit_alpha: bool = True,
) -> float | None:
    """
    Plot MSD(τ) on a log-log axis and optionally fit a power law.

    Returns the fitted exponent α, or None if *fit_alpha* is False.
    """
    ax.plot(tau, msd, color=color, lw=lw, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Lag $\tau$ (s)")
    ax.set_ylabel(r"MSD (m$^2$)")

    if not fit_alpha:
        return None

    mask = (tau > 0) & (msd > 0)
    if mask.sum() < 4:
        return None

    log_tau = np.log(tau[mask])
    log_msd = np.log(msd[mask])
    alpha, log_d = np.polyfit(log_tau, log_msd, 1)

    tau_fit = np.linspace(tau[mask][0], tau[mask][-1], 200)
    ax.plot(tau_fit, np.exp(log_d) * tau_fit ** alpha,
            color=color, lw=0.8, ls="--",
            label=rf"$\alpha$={alpha:.2f}")
    return float(alpha)


def plot_hazard(
    ax: plt.Axes,
    age_centres: np.ndarray,
    hazard: np.ndarray,
    ci_lo: np.ndarray | None = None,
    ci_hi: np.ndarray | None = None,
    *,
    label: str = "",
    color: str = "black",
    fit_inverse_age: bool = True,
    a0: float = A0,
) -> dict | None:
    """
    Plot empirical hazard h(a) and optionally fit h(a) = λ∞ + μ/(a₀+a).

    Returns fitted parameter dict {lambda_inf, mu} or None.
    """
    ax.scatter(age_centres, hazard, color=color, s=20, zorder=3, label=label)

    if ci_lo is not None and ci_hi is not None:
        ax.fill_between(age_centres, ci_lo, ci_hi,
                        alpha=0.2, color=color)

    ax.set_xlabel("Run age $a$ (s)")
    ax.set_ylabel(r"Hazard $h(a)$")

    if not fit_inverse_age:
        return None

    from scipy.optimize import minimize

    def sse(params: np.ndarray) -> float:
        lam_inf, mu = params
        if lam_inf < 0 or mu < 0:
            return 1e9
        h_model = lam_inf + mu / (a0 + age_centres)
        return float(np.sum((hazard - h_model) ** 2))

    res = minimize(sse, x0=[0.02, 1.0], method="Nelder-Mead")
    lam_inf, mu = res.x
    a_fit = np.linspace(age_centres[0], age_centres[-1], 200)
    h_fit = lam_inf + mu / (a0 + a_fit)
    ax.plot(a_fit, h_fit, color=color, lw=1.2, ls="--",
            label=rf"Fit: $\lambda_\infty$={lam_inf:.3f}, $\mu$={mu:.2f}")

    return {"lambda_inf": lam_inf, "mu": mu}


def plot_state_transition_matrix(
    ax: plt.Axes,
    P: np.ndarray,
    states: Sequence[str] = STATE_LABELS,
) -> None:
    """Heatmap of a row-stochastic transition matrix."""
    im = ax.imshow(P, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels([s.capitalize() for s in states])
    ax.set_yticklabels([s.capitalize() for s in states])
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("State transition matrix $P$")

    for i in range(len(states)):
        for j in range(len(states)):
            ax.text(j, i, f"{P[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if P[i, j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Transition prob.")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def save_cache(df: pd.DataFrame, name: str) -> Path:
    """Save *df* as a parquet file in DATA_DIR. Returns the path."""
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path


def cache_path(
    filename: str,
    *,
    cache_name: str = PRIMARY_CACHE_NAME,
    legacy_fallback: bool = True,
) -> Path:
    """Return the canonical processed-cache path, with legacy flat-file fallback."""
    canonical = DATA_DIR / "processed" / cache_name / filename
    if canonical.exists() or not legacy_fallback:
        return canonical
    legacy = DATA_DIR / filename
    if legacy.exists():
        return legacy
    return canonical


def load_cache(name: str, suffix: str | None = None) -> pd.DataFrame:
    """Load a cached parquet artefact from DATA_DIR.

    Parameters
    ----------
    name : str
        Base cache stem, e.g. ``"runs_long"``.
    suffix : str, optional
        Optional cache suffix, e.g. ``"2020_2021_all_teams_pitchfix_sticky_active"``.
        When supplied, loads ``{name}_{suffix}.parquet``.
    """
    stem = f"{name}_{suffix}" if suffix else name
    path = DATA_DIR / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {path}\n"
            "Run 01_data_loading.ipynb first."
        )
    return pd.read_parquet(path)


def save_figure(fig: plt.Figure, name: str, *, tight: bool = True) -> Path:
    """Save figure as both PNG and PDF in FIGURES_DIR."""
    if tight:
        fig.tight_layout()
    base = FIGURES_DIR / name
    for ext in ("pdf", "png"):
        fig.savefig(base.with_suffix(f".{ext}"))
    return base


# ---------------------------------------------------------------------------
# Order state assignment
# ---------------------------------------------------------------------------


def assign_order_state(
    p: pd.Series | np.ndarray,
    bounds: list[float] = STATE_BOUNDS,
    labels: list[str] = STATE_LABELS,
) -> pd.Categorical:
    """
    Bin polarisation values *p* into named order states.

    Parameters
    ----------
    p      : Polarisation values in [0, 1].
    bounds : Bin edges (len = len(labels) + 1).
    labels : State names.
    """
    return pd.cut(p, bins=bounds, labels=labels, include_lowest=True)
