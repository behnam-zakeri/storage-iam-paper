from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.common.config import SCENARIO_COLORS
from src.analysis.data_loader import build_model_markers, load_analysis_data


def build_model_yearly_scatter_tables(
    df_sub,
    *,
    var_x: str,
    var_y: str,
    years,
) -> pd.DataFrame:
    """
    Build one long table for a single model across scenarios and years.
    """
    ts = df_sub.timeseries().reset_index()
    tables = []

    for yr in years:
        if yr not in ts.columns:
            continue

        wide = (
            ts.pivot_table(
                index=["model", "scenario"],
                columns="variable",
                values=yr,
                aggfunc="mean",
            )
            .reset_index()
            .copy()
        )

        if var_x not in wide.columns or var_y not in wide.columns:
            continue

        wide["year"] = yr
        tables.append(wide)

    if not tables:
        return pd.DataFrame(columns=["model", "scenario", "year", var_x, var_y])

    return pd.concat(tables, ignore_index=True)


def plot_single_model_scenario_scatter(
    df_plot: pd.DataFrame,
    *,
    model: str,
    var_x: str,
    var_y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: tuple[float, float] = (0, 1),
    ylim: tuple[float, float] | None = None,
    marker_size: float = 100,
    legend_frameon: bool = False,
):
    """
    Plot one-model scatter across scenarios and years.
    Used for S10-S16 style panels.
    """
    _, eu_nzero, _ = load_analysis_data()
    model_markers = build_model_markers(eu_nzero)

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df_plot.iterrows():
        ax.scatter(
            row[var_x],
            row[var_y],
            marker=model_markers.get(model, "o"),
            color=SCENARIO_COLORS[row["scenario"]],
            s=marker_size,
            edgecolor="grey",
        )

    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=model_markers.get(model, "o"),
            color="w",
            label=model,
            markerfacecolor="gray",
            markeredgecolor="grey",
            markersize=11,
        )
    ]

    shown_scenarios = [s for s in SCENARIO_COLORS if s in set(df_plot["scenario"].unique())]
    for scen in shown_scenarios:
        legend_elements.append(Patch(facecolor=SCENARIO_COLORS[scen], label=scen))

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=legend_frameon,
    )

    return fig, ax

