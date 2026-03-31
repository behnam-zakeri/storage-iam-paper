# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.config import (
    SHEET_FIG2,
    CATEGORY_MAP,
    CATEGORY_ORDER,
    PURPLE,
    FIG2_STEM_LW,
    FIG2_DOT_S,
    MAX_MODELS,
)
from src.common.io_utils import read_excel_sheet


def make_figure2(xlsx_file):
    df0 = read_excel_sheet(xlsx_file, SHEET_FIG2)

    df = df0.loc[:, df0.columns.isin(["Category", "Indicator", "Count"])].copy()
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    df = df.dropna(subset=["Category", "Indicator", "Count"])

    df["Category"] = (
        df["Category"]
        .astype(str)
        .str.strip()
        .apply(lambda s: CATEGORY_MAP.get(s.lower(), s))
    )

    present = [c for c in CATEGORY_ORDER if c in df["Category"].unique()]
    if not present:
        present = list(df["Category"].unique())

    df["Category"] = pd.Categorical(df["Category"], categories=present, ordered=True)
    df = df.sort_values(["Category", "Count", "Indicator"], ascending=[True, False, True])

    # layout
    row_gap = 0.85
    group_gap = 1.30
    header_pad = 0.90

    y_positions = []
    labels = []
    counts = []
    header_positions = []

    y = 0.0
    for cat in present:
        sub = df[df["Category"] == cat]
        if sub.empty:
            continue

        first_y = y
        header_positions.append((first_y - header_pad, cat))

        for _, r in sub.iterrows():
            y_positions.append(y)
            labels.append(str(r["Indicator"]))
            counts.append(float(r["Count"]))
            y += row_gap

        y += group_gap

    y_positions = np.array(y_positions)
    counts = np.array(counts)

    fig_h = max(3.6, 0.26 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(6.4, fig_h), dpi=300)

    # stems
    for yi, xi in zip(y_positions, counts):
        ax.plot(
            [0, xi], [yi, yi],
            color=PURPLE, lw=FIG2_STEM_LW, alpha=0.95,
            solid_capstyle="round", zorder=2
        )

    # dots
    ax.scatter(
        counts, y_positions,
        s=FIG2_DOT_S, c=PURPLE,
        edgecolors="white", linewidths=0.9, zorder=3
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlim(0, MAX_MODELS)
    ax.set_xticks(range(0, MAX_MODELS + 1, 1))
    ax.set_xlabel(f"Number of models (max = {MAX_MODELS})")
    ax.set_title(
        "Model ensemble behaviour across electricity storage indicators",
        pad=15, fontweight="bold"
    )

    # grid
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.18)
    ax.grid(axis="y", visible=False)

    # spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, width=0.8)

    # headers outside plot area
    x_header = -0.70
    for hy, cat in header_positions:
        ax.text(
            x_header, hy, cat,
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=9, fontweight="bold",
            color="#2A2A2A",
            clip_on=False
        )

    plt.subplots_adjust(left=0.40)
    plt.tight_layout()
    return fig, ax
