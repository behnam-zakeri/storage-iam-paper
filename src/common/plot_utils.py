from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mc


def apply_plot_style():
    """
    Shared Matplotlib style for both survey and analysis workflows.
    Uses editable text in vector exports.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def lighten_color(color, amount=0.82):
    """
    Blend a color toward white.

    Parameters
    ----------
    color : matplotlib-compatible color
    amount : float
        0 -> original color
        1 -> white
    """
    c = np.array(mc.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)


def with_alpha(color, alpha: float):
    """
    Return RGBA tuple with specified alpha.
    """
    r, g, b = mc.to_rgb(color)
    return (r, g, b, alpha)


def make_output_base(output_dir, stem: str) -> Path:
    """
    Build base output path without suffix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / stem


def save_figure(
    fig,
    out_base,
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    dpi: int = 600,
):
    """
    Save a figure in one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_base : str | Path
        Path without suffix, e.g. outputs/survey/Figure-3a
    save_png, save_pdf, save_svg : bool
        Output format toggles
    dpi : int
        Raster dpi for PNG
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if save_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")


def remove_top_right_spines(ax):
    """
    Hide top and right spines.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def remove_all_spines(ax):
    """
    Hide all spines.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)


def horizontal_grid_only(ax, linewidth: float = 0.8, alpha: float = 0.35):
    """
    Keep only horizontal grid lines.
    """
    ax.grid(False)
    ax.yaxis.grid(True, linewidth=linewidth, alpha=alpha)
    ax.xaxis.grid(False)


def set_clean_ticks(ax, x_length: float = 3, y_length: float = 0, width: float = 0.8):
    """
    Apply a simple consistent tick style.
    """
    ax.tick_params(axis="x", length=x_length, width=width)
    ax.tick_params(axis="y", length=y_length, width=width)


def scenario_path_legend_handles(
    models: list[str],
    model_markers: dict[str, str],
    scenario_colors: dict[str, str],
    scen_nz: str,
    scen_diag: str,
    include_benchmarks: list | None = None,
):
    """
    Optional helper for building repeated custom legends in analysis plots.
    Returns a list of Line2D handles.

    This helper is intentionally generic and can be extended in analysis modules.
    """
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], linestyle="none", label="Models")]

    for model in sorted(models):
        handles.append(
            Line2D(
                [0], [0],
                marker=model_markers.get(model, "o"),
                color="w",
                label=model,
                markerfacecolor="gray",
                markeredgecolor="grey",
                markersize=7,
            )
        )

    handles.append(Line2D([], [], linestyle="none"))
    handles.append(Line2D([], [], linestyle="none", label="Scenarios"))
    handles.append(Line2D([0], [0], color=scenario_colors[scen_nz], lw=2, label=scen_nz))
    handles.append(
        Line2D(
            [0], [0],
            color=scenario_colors[scen_diag],
            lw=2,
            label=f"{scen_diag} \n(LimBio if missing)",
        )
    )

    if include_benchmarks:
        handles.append(Line2D([], [], linestyle="none"))
        handles.append(Line2D([], [], linestyle="none", label="Benchmarks"))
        handles.extend(include_benchmarks)

    handles.append(Line2D([], [], linestyle="none"))
    handles.append(Line2D([], [], linestyle="none", label="Years"))
    handles.append(
        Line2D(
            [], [],
            linestyle="none",
            label="2030 → 2040",
            marker="o",
            markersize=6,
            markerfacecolor="grey",
            markeredgecolor="grey",
            markeredgewidth=0.8,
        )
    )
    handles.append(
        Line2D(
            [], [],
            linestyle="none",
            label="2050",
            marker="o",
            markersize=6,
            markerfacecolor="grey",
            markeredgecolor="black",
            markeredgewidth=2.0,
        )
    )

    return handles