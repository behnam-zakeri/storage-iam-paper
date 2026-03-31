import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

from config import SHEET_FIG3, MODEL_DISPLAY_SHORT
from src.common.io_utils import read_excel_sheet


def make_figure3b(xlsx_file):
    raw = read_excel_sheet(xlsx_file, SHEET_FIG3)

    # support Criterion / Criteria
    if "Criterion" not in raw.columns:
        if "Criteria" in raw.columns:
            raw = raw.rename(columns={"Criteria": "Criterion"})
        else:
            raise ValueError("Excel sheet must contain a 'Criterion' or 'Criteria' column.")

    if "Dimension" not in raw.columns:
        raise ValueError("Excel sheet must contain a 'Dimension' column.")

    if "Max points" not in raw.columns:
        raise ValueError("Excel sheet must contain a 'Max points' column.")

    meta_cols = ["Dimension", "Criterion", "Max points"]
    model_cols = [c for c in raw.columns if c not in meta_cols]
    models = model_cols

    df_tbl = raw.copy()
    df_tbl["Max points"] = pd.to_numeric(df_tbl["Max points"], errors="coerce")
    for m in model_cols:
        df_tbl[m] = pd.to_numeric(df_tbl[m], errors="coerce")

    norm_vals = df_tbl[model_cols].div(df_tbl["Max points"].replace({0: np.nan}), axis=0)
    norm_mat = norm_vals.to_numpy()

    row_labels = df_tbl["Criterion"].astype(str).tolist()
    is_subtotal_row = [lbl.strip().lower() == "subtotal" for lbl in row_labels]

    # contiguous block ranges by dimension
    block_ranges = []
    start = 0
    for dim, g in df_tbl.groupby("Dimension", sort=False):
        n = len(g)
        block_ranges.append((str(dim), start, start + n - 1))
        start += n

    fig_h = max(6.8, 0.42 * len(row_labels) + 1.8)
    fig = plt.figure(figsize=(12.5, fig_h))

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.15, 2.8, 8.0],
        wspace=0.04
    )

    ax_dim = fig.add_subplot(gs[0, 0])
    ax_lbl = fig.add_subplot(gs[0, 1], sharey=ax_dim)
    ax = fig.add_subplot(gs[0, 2], sharey=ax_dim)

    # heatmap
    cmap = mpl.cm.YlGnBu
    im = ax.imshow(norm_mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    display_models = [MODEL_DISPLAY_SHORT.get(m, m) for m in models]

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(display_models, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(
        axis="x", which="both", top=True, bottom=False,
        labeltop=True, labelbottom=False, length=0, pad=8
    )

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([""] * len(row_labels))
    ax.tick_params(axis="y", length=0)

    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # text values
    for i in range(norm_mat.shape[0]):
        for j in range(norm_mat.shape[1]):
            val = norm_mat[i, j]
            if pd.isna(val):
                txt = ""
                text_color = "black"
            else:
                txt = f"{val:.2f}".rstrip("0").rstrip(".")
                text_color = "white" if val >= 0.58 else "black"

            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8.2,
                fontweight="bold" if is_subtotal_row[i] else "normal",
                color=text_color
            )

    # subtotal row outlines
    for i, is_sub in enumerate(is_subtotal_row):
        if is_sub:
            ax.add_patch(Rectangle(
                (-0.5, i - 0.5), len(models), 1,
                fill=False, edgecolor="0.20", linewidth=1.4
            ))

    # separators between dimensions
    for _, _, y1 in block_ranges[:-1]:
        ax.hlines(y1 + 0.5, -0.5, len(models) - 0.5, colors="0.25", linewidth=1.8)

    # criterion label axis
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(len(row_labels) - 0.5, -0.5)
    ax_lbl.axis("off")

    for i, label in enumerate(row_labels):
        ax_lbl.text(
            0.98, i, label,
            ha="right", va="center",
            fontsize=9.2,
            fontweight="bold" if is_subtotal_row[i] else "normal",
            color="0.15"
        )

    for _, _, y1 in block_ranges[:-1]:
        ax_lbl.hlines(y1 + 0.5, 0.0, 1.0, colors="0.88", linewidth=1.0)

    # dimension strip axis
    ax_dim.set_xlim(0, 1)
    ax_dim.set_ylim(len(row_labels) - 0.5, -0.5)
    ax_dim.axis("off")

    for dim, y0, y1 in block_ranges:
        ax_dim.text(
            1.50, (y0 + y1) / 2, str(dim),
            rotation=90,
            ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="0.25",
            linespacing=1.15
        )

    fig.suptitle(
        "b) Criterion-level diagnostic heatmap of the Storage Representation Framework",
        x=0.55, y=0.96, fontsize=12, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalized score", fontsize=9.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)

    plt.tight_layout(rect=[0.02, 0.03, 0.96, 0.94])
    return fig

