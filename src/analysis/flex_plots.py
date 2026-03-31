from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from src.common.config import ANALYSIS_OUTPUT_DIR, SELECTED_REGION, SELECTED_SCENARIO
from src.common.plot_utils import save_figure
from src.analysis.flex_metrics import get_storage_metric_column


def _coerce_indicator_values(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Coerce requested indicator columns to numeric and clip to [0, 1] for plotting.
    """
    out = df.copy()
    for col in indicators:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0, upper=1)
    return out


def _prepare_flex_plot_data(
    df: pd.DataFrame,
    indicators: list[str],
    sort_by: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Prepare and optionally sort the flexibility-indicator table.
    """
    out = _coerce_indicator_values(df, indicators)

    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)
    else:
        out = out.sort_index()

    return out


def plot_minibars_portfolio_clean(
    df: pd.DataFrame,
    indicators: list[str],
    titles: list[str],
    demand_col: str = "r_flexible_demand_proxy",
    tier_col: str = "demand_response",
    sort_by: str | None = None,
    ascending: bool = False,
    base_color: str = "mediumorchid",
    track_alpha: float = 0.05,
    edge_alpha: float = 0.30,
    bar_height: float = 0.50,
    show_scale_text: bool = True,
    outfile_stem: str | None = None,
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    title_prefix: str = "c) Electricity storage & flexibility portfolio",
    subtitle: str = "(EU27 & UK, NetZero scenario, 2050)",
    faint_bars: bool = False,
):
    """
    Clean minibar 'bar-table' figure.

    Features preserved from the original script:
    - one column per flexibility indicator
    - faint full-width background bars if requested
    - shaded demand-flexibility column according to demand-response tier
    - model names on rows
    - title + subtitle above plot
    - horizontal rules above and below bar area
    - scale box below the figure
    """
    if len(indicators) != len(titles):
        raise ValueError("'indicators' and 'titles' must have the same length.")

    d = _prepare_flex_plot_data(
        df=df,
        indicators=indicators,
        sort_by=sort_by,
        ascending=ascending,
    )

    models = d.index.astype(str).tolist()
    n_models = len(models)
    n_cols = len(indicators)
    y = np.arange(n_models)

    # Demand-response tier intensity map
    alpha_map = {
        "No": 0.0,
        "Proxy": 0.25,
        "Low": 0.55,
        "Medium": 0.55,
        "High": 1.0,
        "Unknown": 0.25,
    }

    if tier_col in d.columns:
        tiers = d[tier_col].astype(str).fillna("Unknown").values
    else:
        tiers = np.array(["Unknown"] * n_models)

    # figure sizing
    fig_w = 1.25 * n_cols + 2.2
    fig_h = max(2.2, 0.22 * n_models)

    fig, axes = plt.subplots(
        1,
        n_cols,
        sharey=True,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.04},
    )
    if n_cols == 1:
        axes = [axes]

    base_rgb = to_rgba(base_color, 1.0)[:3]
    track_rgba = (*base_rgb, track_alpha)
    edge_rgba = (0, 0, 0, edge_alpha)

    for j, ax in enumerate(axes):
        col = indicators[j]
        vals = d[col].values

        if faint_bars:
            ax.barh(
                y,
                np.ones_like(vals),
                height=bar_height,
                color=track_rgba,
                edgecolor=edge_rgba,
                linewidth=0.2,
            )

        if col == demand_col:
            fill_colors = [(*base_rgb, alpha_map.get(t, 0.25)) for t in tiers]
        else:
            fill_colors = [(*base_rgb, 1.0)] * n_models

        ax.barh(
            y,
            vals,
            height=bar_height,
            color=fill_colors,
            edgecolor="none",
        )

        ax.set_xlim(0, 0.5)
        ax.set_title(titles[j], fontsize=9, pad=10)

        ax.grid(False)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

        ax.set_xticks([])
        ax.tick_params(axis="y", length=0)

        if j != 0:
            ax.set_yticklabels([])

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models, fontsize=8)
    axes[0].invert_yaxis()

    # Title area
    ax0_pos = axes[0].get_position()
    title_y = ax0_pos.y1 + 0.10
    fig.text(ax0_pos.x0 - 0.05, title_y, "Model", ha="left", va="bottom", fontsize=9)

    fig.suptitle(
        title_prefix,
        fontsize=11,
        y=1.30,
        fontweight="bold",
        ha="center",
    )
    fig.text(
        0.5,
        1.13,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
    )

    # Horizontal rules
    left = axes[0].get_position().x0 - 0.12
    right = axes[-1].get_position().x1
    y_rule_top = ax0_pos.y1 + 0.006

    fig.add_artist(
        Line2D(
            [left, right],
            [y_rule_top, y_rule_top],
            transform=fig.transFigure,
            color="grey",
            lw=0.6,
            alpha=0.8,
        )
    )

    bottom_axes = min(ax.get_position().y0 for ax in axes)
    y_rule_bottom = bottom_axes - 0.015
    fig.add_artist(
        Line2D(
            [left, right],
            [y_rule_bottom, y_rule_bottom],
            transform=fig.transFigure,
            color="grey",
            lw=0.6,
            alpha=0.8,
        )
    )

    # Scale box
    if show_scale_text:
        scale_w = (right - left) * 0.143
        scale_h = 0.05
        scale_x = left + (right - left - scale_w) / 2
        scale_y = y_rule_bottom - 0.075

        rect = Rectangle(
            (scale_x, scale_y),
            scale_w,
            scale_h,
            transform=fig.transFigure,
            fill=False,
            edgecolor="grey",
            linewidth=0.7,
        )
        fig.add_artist(rect)

        ticks = [0.0, 0.5, 1.0]
        tick_labels = ["0", "0.25", "0.5"]
        for t, lab in zip(ticks, tick_labels):
            tx = scale_x + t * scale_w
            fig.text(tx, scale_y - 0.012, lab, ha="center", va="top", fontsize=7)

        fig.text(
            scale_x - 0.01,
            scale_y + scale_h / 2,
            "Scale",
            ha="right",
            va="center",
            fontsize=7,
        )

    if outfile_stem:
        save_figure(
            fig,
            ANALYSIS_OUTPUT_DIR / outfile_stem,
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, axes


def build_figure4c_inputs(
    indices: pd.DataFrame,
    storage_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Build the default indicator list and display titles for Figure 4c.
    """
    if storage_col is None:
        storage_col = get_storage_metric_column(indices)

    indicators = [
        storage_col,
        "r_flexible_gen_over_total_electricity",
        "r_flexible_demand_proxy",
        "r_h2_electrolyzer_capacity_over_total_elec_capacity",
        "r_spatial",
        "r_curt",
    ]
    titles = [
        "Electricity\nstorage",
        "Flexible\ngeneration",
        "Flexible\ndemand",
        "Hydrogen\nelectrolysis",
        "Spatial\nflexibility",
        "VRE\ncurtailment",
    ]
    return indicators, titles


def make_figure4c(
    indices: pd.DataFrame,
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    faint_bars: bool = False,
    sort_by: str | None = "model",
    ascending: bool = False,
    base_color: str = "orchid",
):
    """
    Manuscript Figure 4c:
    Electricity storage & flexibility portfolio.

    Parameters
    ----------
    indices : DataFrame
        Output from compute_flex_ratios_all().
    """
    storage_col = get_storage_metric_column(indices)
    indicators, titles = build_figure4c_inputs(indices, storage_col=storage_col)

    # Subtitle from data when available
    region = str(indices["region"].iloc[0]) if "region" in indices.columns and not indices.empty else SELECTED_REGION
    scenario = str(indices["scenario"].iloc[0]) if "scenario" in indices.columns and not indices.empty else SELECTED_SCENARIO
    year = str(indices["year"].iloc[0]) if "year" in indices.columns and not indices.empty else "2050"
    subtitle = f"({region.replace('(*)', '').strip()}, {scenario} scenario, {year})"

    fig, axes = plot_minibars_portfolio_clean(
        df=indices,
        indicators=indicators,
        titles=titles,
        demand_col="r_flexible_demand_proxy",
        tier_col="demand_response",
        sort_by=sort_by,
        ascending=ascending,
        base_color=base_color,
        faint_bars=faint_bars,
        outfile_stem="Figure-4c",
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
        title_prefix="c) Electricity storage & flexibility portfolio",
        subtitle=subtitle,
    )

    # Source data: keep figure columns plus diagnostics useful for interpretation
    keep_cols = [
        "scenario",
        "region",
        "year",
        "demand_response",
        "spatial_resolution",
        storage_col,
        "r_flexible_gen_over_total_electricity",
        "r_flexible_demand_proxy",
        "r_h2_electrolyzer_capacity_over_total_elec_capacity",
        "r_net_trade_intensity",
        "r_spatial_resolution",
        "r_spatial",
        "r_curt",
        "h2_electrolyzer_capacity_gw_reported",
        "h2_electrolyzer_capacity_gw_inferred",
        "h2_electrolyzer_capacity_gw_used",
        "h2_capacity_was_inferred",
    ]
    keep_cols = [c for c in keep_cols if c in indices.columns]

    source_data = indices.reset_index().copy()
    source_data = source_data[["model"] + keep_cols]

    return fig, source_data