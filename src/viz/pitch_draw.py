"""
Pitch drawing utilities for 2D visualisation.
"""

from __future__ import annotations
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.axes

import matplotlib.patches as patches
import matplotlib.axes
from typing import List, Tuple

def draw_pitch(
    ax: matplotlib.axes.Axes,
    pitch_xy: List[Tuple[float, float]],
    margin_m: float = 6.0,
    show_outer_box: bool = True,
    active_depth_m: float | None = None,  # metres from lines defining the active zone
    show_active_zone: bool = True,
    show_scale_bar: bool = True,
    scale_bar_length_m: float = 10.0,  # length of the scale bar in metres
    # NEW: fill the region between pitch bounds and axes (outer box)
    show_outer_fill: bool = True,
    outer_fill_color: str = "gray",
    outer_fill_alpha: float = 0.4,
    # NEW: base pitch fill (useful when outer fill is on)
    pitch_fill_color: str = "white",
) -> None:
    """
    Draw a football pitch from pitch_xy plus:
      - outer rectangle at the simulation limits (pitch + margin_m)
      - optional filled outer region (between pitch bounds and figure/axes limits)
      - red ring (bench strip) between pitch perimeter and active zone
      - light-green filled active zone (depth >= active_depth_m)
      - horizontal scale bar (in metres).

    Pitch lines are always drawn on top of the coloured areas.

    Notes
    -----
    - When show_outer_fill=True, we first paint the whole outer box with outer_fill_color,
      then paint the pitch interior with pitch_fill_color. This creates a clean "ring"
      between the pitch perimeter and the figure limits.
    """
    import numpy as np

    P = np.asarray(pitch_xy, dtype=float)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)

    # outer simulation box
    outer_xmin = xmin - margin_m
    outer_xmax = xmax + margin_m
    outer_ymin = ymin - margin_m
    outer_ymax = ymax + margin_m

    # Optional: fill the margin/background ring
    if show_outer_fill:
        # Fill entire outer box
        ax.add_patch(
            patches.Rectangle(
                (outer_xmin, outer_ymin),
                outer_xmax - outer_xmin,
                outer_ymax - outer_ymin,
                facecolor=outer_fill_color,
                edgecolor="none",
                alpha=outer_fill_alpha,
                zorder=0.15,
            )
        )
        # Paint pitch interior back to a base colour so only the margin shows outer_fill_color
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=pitch_fill_color,
                edgecolor="none",
                alpha=1.0,
                zorder=0.25,
            )
        )

    if show_outer_box:
        ax.add_patch(
            patches.Rectangle(
                (outer_xmin, outer_ymin),
                outer_xmax - outer_xmin,
                outer_ymax - outer_ymin,
                fill=False,
                ec="black",
                lw=2.5,
                ls="-",
                zorder=0.5,
            )
        )

    # coloured pitch areas (bench strip + active zone)
    if show_active_zone and active_depth_m is not None and active_depth_m > 0:
        pitch_width = xmax - xmin
        pitch_height = ymax - ymin

        # entire pitch light red (bench strip background)
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),
                pitch_width,
                pitch_height,
                facecolor="red",
                edgecolor="none",
                alpha=0.5,
                zorder=0.6,
            )
        )

        inner_xmin = xmin + active_depth_m
        inner_xmax = xmax - active_depth_m
        inner_ymin = ymin + active_depth_m
        inner_ymax = ymax - active_depth_m

        if inner_xmax > inner_xmin and inner_ymax > inner_ymin:
            ax.add_patch(
                patches.Rectangle(
                    (inner_xmin, inner_ymin),
                    inner_xmax - inner_xmin,
                    inner_ymax - inner_ymin,
                    facecolor="springgreen",
                    edgecolor="none",
                    alpha=1,
                    zorder=0.7,
                )
            )

    # pitch lines
    ax.add_patch(
        patches.Polygon(P, closed=True, fill=False, lw=2.0, ec="black", zorder=1.5)
    )

    midx = 0.5 * (xmin + xmax)
    midy = 0.5 * (ymin + ymax)

    # halfway line
    ax.plot([midx, midx], [ymin, ymax], lw=1.2, color="black", zorder=1.5)

    # centre circle + spot
    ax.add_patch(
        patches.Circle((midx, midy), 9.15, fill=False, ec="black", lw=1.0, zorder=1.5)
    )
    ax.add_patch(patches.Circle((midx, midy), 0.2, color="black", zorder=1.5))

    # penalty & goal boxes (assume goals at xmin/xmax)
    for side in ("left", "right"):
        gx0 = xmin if side == "left" else xmax
        sgn = +1 if side == "left" else -1

        # penalty area
        ax.add_patch(
            patches.Rectangle(
                (min(gx0, gx0 + sgn * 16.5), midy - 40.32 / 2),
                16.5,
                40.32,
                fill=False,
                ec="black",
                lw=1.0,
                zorder=1.5,
            )
        )

        # 6-yard box
        ax.add_patch(
            patches.Rectangle(
                (min(gx0, gx0 + sgn * 5.5), midy - 18.32 / 2),
                5.5,
                18.32,
                fill=False,
                ec="black",
                lw=1.0,
                zorder=1.5,
            )
        )

        # penalty spot
        ax.add_patch(
            patches.Circle((gx0 + sgn * 11.0, midy), 0.2, color="black", zorder=1.5)
        )

    # scale bar
    if show_scale_bar and scale_bar_length_m is not None and scale_bar_length_m > 0:
        pitch_width = xmax - xmin
        length = min(scale_bar_length_m, 0.8 * pitch_width)

        bar_y = outer_ymin + 0.5 * (ymin - outer_ymin)
        bar_x_center = 0.5 * (xmin + xmax)
        bar_x0 = bar_x_center - length / 2.0
        bar_x1 = bar_x_center + length / 2.0

        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], lw=2.0, color="black", zorder=2)
        tick_h = (ymin - outer_ymin) * 0.08
        ax.plot([bar_x0, bar_x0], [bar_y - tick_h, bar_y + tick_h], lw=1.5, color="black", zorder=2)
        ax.plot([bar_x1, bar_x1], [bar_y - tick_h, bar_y + tick_h], lw=1.5, color="black", zorder=2)

        ax.text(
            bar_x_center,
            bar_y - 1.5 * tick_h,
            f"{int(round(length))} m",
            ha="center",
            va="top",
            fontsize=9,
            zorder=2,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(outer_xmin, outer_xmax)
    ax.set_ylim(outer_ymin, outer_ymax)

    # Keep default axes background white; the outer-fill patch provides the colour when enabled.
    ax.set_facecolor("white")

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
