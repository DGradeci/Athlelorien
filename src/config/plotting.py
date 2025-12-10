"""
Plotting configuration utilities.

Usage
-----
from config.plotting import configure_plotting
configure_plotting()
"""

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


def configure_plotting(base: int = 12) -> None:
    """
    Configure global Matplotlib settings for a clean, publication-style look.

    Parameters
    ----------
    base : int
        Base font size. Other font sizes are scaled from this.
    """
    # Reset to Matplotlib defaults (wipes seaborn / Jupyter styles)
    plt.style.use("default")

    # If seaborn has been used earlier in the session, reset it as well
    try:
        import seaborn as sns  # type: ignore

        sns.reset_orig()
    except Exception:
        pass

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern", "STIX", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Font sizes
            "font.size": base,
            "axes.titlesize": base * 1.2,
            "axes.labelsize": base,
            "xtick.labelsize": base * 0.9,
            "ytick.labelsize": base * 0.9,
            "legend.fontsize": base * 0.9,
            "legend.title_fontsize": base,
            "figure.titlesize": base * 1.3,
            # Colours – lock everything to solid black
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            # Solid white backgrounds (no transparency surprises in exports)
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
