import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerPatch

from src.common.config import SHEET_S1, S1_SEGMENTS
from src.common.io_utils import read_excel_sheet


def make_figure_s1(xlsx_file):
    tbl = read_excel_sheet(xlsx_file, SHEET_S1)

    row_label_col = tbl.columns[0]
    tbl = tbl.rename(columns={row_label_col: "Metric"}).set_index("Metric")

    max_col = "Max points"
    if max_col not in tbl.columns:
        raise ValueError("Sheet 'SRF-total' must contain a 'Max points' column.")

    model_cols = [c for c in tbl.columns if c != max_col]
    models = model_cols

    def get_row(metric_contains: str):
        m = tbl.index.astype(str).str.lower().str.contains(metric_contains.lower())
        if not m.any():
            raise ValueError(
                f"Could not find a row containing '{metric_contains}' in sheet 'SRF-total'."
            )
        return tbl.loc[m].iloc[0]

    storage_s = get_row("Storage representation")
    vre_s     = get_row("VRE modelling")
    temp_s    = get_row("Temporal representation")
    spatial_s = get_row("Spatial representation")

    storage = storage_s[models].to_numpy(dtype=float)
    vre     = vre_s[models].to_numpy(dtype=float)
    temp    = temp_s[models].to_numpy(dtype=float)
    spatial = spatial_s[models].to_numpy(dtype=float)

    max_storage = float(storage_s[max_col])
    max_vre     = float(vre_s[max_col])
    max_temp    = float(temp_s[max_col])
    max_spatial = float(spatial_s[max_col])

    x = storage / max_storage
    y = vre / max_vre
    t = temp / max_temp
    s = spatial / max_spatial

    n_segments = S1_SEGMENTS
    filled = np.clip(np.rint(s * n_segments).astype(int), 0, n_segments)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "orchid_shades",
        ["#F7E6FA", "#E8B9F2", "#DA70D6", "#B04BB7", "#6A1B6F"]
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    theta_step = 360 / n_segments

    def seg_angles(k):
        th2 = 90 - k * theta_step
        th1 = 90 - (k + 1) * theta_step
        return th1, th2

    fig, ax = plt.subplots(figsize=(11.8, 6.6), dpi=170)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(right=0.70)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    bubble_size = 360
    sc = ax.scatter(
        x, y,
        s=bubble_size,
        c=t,
        cmap=cmap,
        norm=norm,
        edgecolor="#2B2B2B",
        linewidth=0.8,
        zorder=3
    )

    ring_outer = 0.05
    ring_width = 0.015
    ring_outline = "#D7D7D7"
    ring_fill = "#2E2E2E"

    patches, colors = [], []
    for xi, yi, f in zip(x, y, filled):
        for k in range(n_segments):
            th1, th2 = seg_angles(k)
            patches.append(Wedge((xi, yi), r=ring_outer, theta1=th1, theta2=th2, width=ring_width))
            colors.append(ring_outline)
            if k < f:
                patches.append(Wedge((xi, yi), r=ring_outer, theta1=th1, theta2=th2, width=ring_width))
                colors.append(ring_fill)

    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="none", zorder=4))

    ax.set_xlabel("Explicitness in modelling electricity storage", labelpad=20, fontweight="bold")
    ax.set_ylabel("Breadth in VRE modelling", labelpad=20, fontweight="bold")
    ax.set_title("Electricity Storage Representation Framework (SRF)", pad=15, fontweight="bold")

    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_thirds = [
        xmin + (xmax - xmin) / 3,
        xmin + 2 * (xmax - xmin) / 3,
    ]
    y_thirds = [
        ymin + (ymax - ymin) / 3,
        ymin + 2 * (ymax - ymin) / 3,
    ]
    ax.set_axisbelow(False)

    for xv in x_thirds:
        ax.axvline(xv, color="#6D3BB8", linestyle="-", linewidth=0.9, alpha=0.25, zorder=2)

    for yv in y_thirds:
        ax.axhline(yv, color="#6D3BB8", linestyle="-", linewidth=0.9, alpha=0.25, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    x_centers = [
        xmin + (xmax - xmin) / 6,
        xmin + (xmax - xmin) / 2,
        xmin + 5 * (xmax - xmin) / 6,
    ]
    y_centers = [
        ymin + (ymax - ymin) / 6,
        ymin + (ymax - ymin) / 2,
        ymin + 5 * (ymax - ymin) / 6,
    ]

    x_labels = ["Implicit", "Middle", "Explicit"]
    y_labels = ["Reduced-form", "Middle", "Detailed"]

    for xc, lab in zip(x_centers, x_labels):
        ax.text(
            xc, -0.01, lab,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=9, color="#4A4A4A", style="italic",
        )

    for yc, lab in zip(y_centers, y_labels):
        ax.text(
            -0.01, yc, lab,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", rotation=90,
            fontsize=9, color="#4A4A4A", style="italic",
        )

    label_xy = {
        "AIM-T": (x[0] + 0.04, y[0] + 0.03),
        "GCAM": (x[1] + 0.04, y[1] + 0.03),
        "IMAGE": (x[2] + 0.04, y[2] + 0.03),
        "MESSAGEix": (x[3] + 0.04, y[3] + 0.03),
        "PROMETHEUS": (x[4] + 0.04, y[4] + 0.03),
        "REMIND": (x[5] + 0.04, y[5] + 0.03),
        "TIAM-ECN": (x[6] + 0.04, y[6] + 0.02),
        "WITCH": (x[7] + 0.04, y[7] - 0.04),
    }

    for name, xi, yi in zip(models, x, y):
        lx, ly = label_xy.get(name, (xi + 0.03, yi + 0.02))
        ax.text(lx, ly, name, fontsize=8, color="#1A1A1A", zorder=6)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_third = xmin + (xmax - xmin) / 3.0
    y_third = ymin + (ymax - ymin) / 3.0

    arrow_color = "#4A4A4A"
    arrow_lw = 0.9

    y_above = -0.02
    ax.annotate(
        "", xy=(0.2, y_above), xytext=(1.3 * x_third, y_above),
        xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="->", lw=arrow_lw, color=arrow_color),
        annotation_clip=False
    )
    ax.annotate(
        "", xy=(0.8, y_above), xytext=(1.85 * x_third, y_above),
        xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="->", lw=arrow_lw, color=arrow_color),
        annotation_clip=False
    )

    x_left = -0.02
    ax.annotate(
        "", xy=(x_left, 0.24), xytext=(x_left, 1.3 * y_third),
        xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="->", lw=arrow_lw, color=arrow_color),
        annotation_clip=False
    )
    ax.annotate(
        "", xy=(x_left, 0.8), xytext=(x_left, 1.85 * y_third),
        xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="->", lw=arrow_lw, color=arrow_color),
        annotation_clip=False
    )

    legend_face = "white"
    legend_edge = "#D0D0D0"

    cax = fig.add_axes([0.65, 0.56, 0.115, 0.03])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Low-", "Med-", "High-detail"])
    cbar.ax.tick_params(length=0, labelsize=8)
    cbar.ax.set_title("Temporal representation", fontsize=8, pad=4, fontweight="bold")

    for sp in cax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(legend_edge)
        sp.set_linewidth(0.8)
    cax.set_facecolor(legend_face)

    def ring_proxy(filled_segments, label):
        wedges = []
        for k in range(n_segments):
            th1, th2 = seg_angles(k)
            wedges.append(Wedge((0, 0), 1.0, th1, th2, width=0.35, facecolor=ring_outline, edgecolor="none"))
            if k < filled_segments:
                wedges.append(Wedge((0, 0), 1.0, th1, th2, width=0.35, facecolor=ring_fill, edgecolor="none"))
        return Circle((0, 0), radius=1.0, label=label), wedges

    class HandlerRing(HandlerPatch):
        def __init__(self, wedges, **kwargs):
            super().__init__(**kwargs)
            self._wedges = wedges

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            cx0 = xdescent + width / 2
            cy0 = ydescent + height / 2
            r = min(width, height) / 1 * 0.95
            artists = []
            for w in self._wedges:
                ww = Wedge(
                    (cx0, cy0), r=r,
                    theta1=w.theta1, theta2=w.theta2,
                    width=r * 0.35,
                    facecolor=w.get_facecolor(),
                    edgecolor="none"
                )
                ww.set_transform(trans)
                artists.append(ww)
            return artists

    handles, handler_map = [], {}
    for segs, lab in [(2, "Low-"), (4, "Med-"), (6, "High-resolution")]:
        circ, wedges = ring_proxy(segs, lab)
        handles.append(circ)
        handler_map[circ] = HandlerRing(wedges)

    ring_leg = ax.legend(
        handles=handles,
        title="Spatial representation",
        loc="upper left",
        bbox_to_anchor=(1.04, 0.52),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.6,
        labelspacing=0.6,
        handlelength=1.5,
        handler_map=handler_map
    )
    ring_leg.get_frame().set_facecolor(legend_face)
    ring_leg.get_frame().set_edgecolor(legend_edge)
    ring_leg.get_frame().set_linewidth(0.8)
    ring_leg.get_title().set_fontweight("bold")

    return fig