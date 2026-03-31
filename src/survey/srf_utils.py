# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge, Circle

from config import DIMENSION_SPECS, FIG3A_LAYOUT


def short_indicator_label(text):
    mapping = {
        "Storage technology": "S1",
        "Storage parameters": "S2",
        "Storage services": "S3",
        "Storage integration": "S4",
        "VRE technology": "V1",
        "VRE parameters": "V2",
        "VRE integration": "V3",
        "Flexibility portfolio": "V4",
        "Temporal resolution": "T1",
        "Temporal modelling": "T2",
        "Spatial resolution": "Sp1",
        "Grid representation": "Sp2",
    }
    return mapping.get(text, text.replace(" ", "\n", 1))


def build_long_srf(df_raw, dimension_col, criterion_col, max_col, model_cols):
    df = df_raw.melt(
        id_vars=[dimension_col, criterion_col, max_col],
        value_vars=model_cols,
        var_name="model",
        value_name="score"
    )
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df[max_col] = pd.to_numeric(df[max_col], errors="coerce")
    df["norm"] = df["score"] / df[max_col]
    return df


def attach_criteria_order(df_use, dimension_col, criterion_col, dimension_specs=None):
    """
    Returns a copy of dimension_specs with Excel-order criteria lists attached.
    """
    if dimension_specs is None:
        dimension_specs = DIMENSION_SPECS

    specs = {k: v.copy() for k, v in dimension_specs.items()}

    criteria_order = (
        df_use[[dimension_col, criterion_col]]
        .drop_duplicates()
        .groupby(dimension_col)[criterion_col]
        .apply(list)
        .to_dict()
    )

    for dim in specs:
        specs[dim]["criteria_order"] = criteria_order.get(dim, [])

    return specs


def draw_segmented_glyph(
    ax,
    sub_df,
    dimension_specs,
    lighten_color,
    show_dim_labels=False,
    show_criterion_labels=False,
    show_dim_labels_guide=False
):
    r_max = FIG3A_LAYOUT["r_max"]
    r_mid = FIG3A_LAYOUT["r_mid"]
    inner_hole = FIG3A_LAYOUT["inner_hole"]
    sep_deg = FIG3A_LAYOUT["sep_deg"]

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.62, 1.62)
    ax.set_aspect("equal")
    ax.axis("off")

    # guide circles
    for rr, lw, col in [(r_mid, 0.9, "0.82"), (r_max, 1.0, "0.78")]:
        ax.add_patch(Circle((0, 0), rr, facecolor="none", edgecolor=col, linewidth=lw, zorder=0))

    # quadrant guide lines
    ax.plot([-1.20, 1.20], [0, 0], color="0.84", lw=0.75, zorder=0)
    ax.plot([0, 0], [-1.20, 1.20], color="0.84", lw=0.75, zorder=0)

    # quadrant backgrounds
    for dim, spec in dimension_specs.items():
        ax.add_patch(Wedge(
            center=(0, 0),
            r=r_max,
            theta1=spec["theta1"],
            theta2=spec["theta2"],
            width=r_max - inner_hole,
            facecolor=lighten_color(spec["color"], 0.90),
            edgecolor="white",
            linewidth=1.0,
            alpha=0.14,
        ))

    # segmented wedges
    for dim, spec in dimension_specs.items():
        crits = spec.get("criteria_order", [])
        if len(crits) == 0:
            continue

        q1, q2 = spec["theta1"], spec["theta2"]
        total_span = q2 - q1
        seg_span = total_span / len(crits)

        for i, crit in enumerate(crits):
            frac = sub_df.loc[sub_df["criterion"] == crit, "norm"]
            frac = float(frac.iloc[0]) if len(frac) else 0.0

            t1 = q1 + i * seg_span + sep_deg / 2
            t2 = q1 + (i + 1) * seg_span - sep_deg / 2

            # background sector
            ax.add_patch(Wedge(
                center=(0, 0),
                r=r_max,
                theta1=t1,
                theta2=t2,
                width=r_max - inner_hole,
                facecolor=lighten_color(spec["color"], 0.93),
                edgecolor="white",
                linewidth=0.9,
                alpha=0.10,
            ))

            # filled sector
            if frac > 0:
                r_fill = inner_hole + frac * (r_max - inner_hole)
                ax.add_patch(Wedge(
                    center=(0, 0),
                    r=r_fill,
                    theta1=t1,
                    theta2=t2,
                    width=r_fill - inner_hole,
                    facecolor=spec["color"],
                    edgecolor="white",
                    linewidth=0.9,
                    alpha=0.94,
                ))

            # criterion labels for guide
            if show_criterion_labels:
                ang = np.deg2rad((t1 + t2) / 2.0)
                r_lab = 1.18
                x, y = r_lab * np.cos(ang), r_lab * np.sin(ang)

                label = short_indicator_label(crit)
                ha = "left" if x > 0.15 else "right" if x < -0.15 else "center"
                va = "bottom" if y > 0.15 else "top" if y < -0.15 else "center"

                ax.text(
                    x, y, label,
                    fontsize=8, ha=ha, va=va,
                    color=spec["color"], linespacing=1.0
                )

    # center hole
    ax.add_patch(Circle((0, 0), inner_hole - 0.01, facecolor="white", edgecolor="none"))

    # radial scale labels only on guide
    if show_criterion_labels:
        ax.text(0.04, r_mid + 0.01, "0.5", fontsize=8, color="0.42", ha="left", va="bottom")
        ax.text(0.04, r_max + 0.01, "1", fontsize=8, color="0.42", ha="left", va="bottom")

    # dimension labels: full labels for guide
    if show_dim_labels_guide:
        ax.text(0, 1.43, "Storage", ha="center", va="bottom", fontsize=10,
                color=dimension_specs["Storage representation"]["color"])
        ax.text(1.34, 0, "VRE", ha="left", va="center", fontsize=10,
                color=dimension_specs["VRE integration"]["color"])
        ax.text(0, -1.36, "Temporal", ha="center", va="top", fontsize=10,
                color=dimension_specs["Temporal representation"]["color"])
        ax.text(-1.34, 0, "Spatial", ha="right", va="center", fontsize=10,
                color=dimension_specs["Spatial representation"]["color"])

    # dimension labels: short model labels
    if show_dim_labels:
        ax.text(0, 1.23, "S", ha="center", va="bottom", fontsize=10,
                color=dimension_specs["Storage representation"]["color"])
        ax.text(1.23, 0, "V", ha="left", va="center", fontsize=10,
                color=dimension_specs["VRE integration"]["color"])
        ax.text(0, -1.23, "T", ha="center", va="top", fontsize=10,
                color=dimension_specs["Temporal representation"]["color"])
        ax.text(-1.23, 0, "Sp", ha="right", va="center", fontsize=10,
                color=dimension_specs["Spatial representation"]["color"])


def build_guide_df(dimension_specs):
    guide_vals = {
        "Storage technology": 0.85,
        "Storage parameters": 0.25,
        "Storage services": 0.55,
        "Storage integration": 0.35,
        "VRE technology": 0.90,
        "VRE parameters": 0.65,
        "VRE integration": 0.45,
        "Flexibility portfolio": 0.70,
        "Temporal resolution": 0.50,
        "Temporal modelling": 0.25,
        "Spatial resolution": 0.60,
        "Grid representation": 0.35,
    }

    guide_rows = []
    for dim, spec in dimension_specs.items():
        for crit in spec.get("criteria_order", []):
            guide_rows.append({
                "dimension": dim,
                "criterion": crit,
                "norm": guide_vals.get(crit, 0.65),
            })

    return pd.DataFrame(guide_rows)

