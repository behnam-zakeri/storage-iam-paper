from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.common.config import ANALYSIS_OUTPUT_DIR, SELECTED_REGION, SELECTED_SCENARIO
from src.common.plot_utils import save_figure
from src.analysis.data_loader import load_analysis_data


POWER_CAPACITY_VARIABLES = [
    "Capacity|Electricity|Coal|w/o CCS",
    "Capacity|Electricity|Coal|w/ CCS",
    "Capacity|Electricity|Gas|w/o CCS",
    "Capacity|Electricity|Gas|w/ CCS",
    "Capacity|Electricity|Biomass|w/o CCS",
    "Capacity|Electricity|Biomass|w/ CCS",
    "Capacity|Electricity|Nuclear",
    "Capacity|Electricity|Other",
    "Capacity|Electricity|Geothermal",
    "Capacity|Electricity|Hydro",
    "Capacity|Electricity|Solar",
    "Capacity|Electricity|Wind",
    "Capacity|Electricity|Storage|Power",
    "Capacity|Electricity|Storage|Power_new",
]


POWER_CAPACITY_LABELS = {
    "Capacity|Electricity|Coal|w/o CCS": "Coal",
    "Capacity|Electricity|Coal|w/ CCS": "Coal w/ CCS",
    "Capacity|Electricity|Gas|w/o CCS": "Gas",
    "Capacity|Electricity|Gas|w/ CCS": "Gas w/ CCS",
    "Capacity|Electricity|Biomass|w/o CCS": "Biomass",
    "Capacity|Electricity|Biomass|w/ CCS": "Biomass w/ CCS",
    "Capacity|Electricity|Nuclear": "Nuclear",
    "Capacity|Electricity|Other": "Other",
    "Capacity|Electricity|Geothermal": "Geothermal",
    "Capacity|Electricity|Hydro": "Hydro",
    "Capacity|Electricity|Solar": "Solar",
    "Capacity|Electricity|Wind": "Wind",
    "Capacity|Electricity|Storage|Power": "Storage",
    "Capacity|Electricity|Storage|Power_new": "Storage",
}


POWER_CAPACITY_COLORS = {
    "Coal": "#4D4D4D",
    "Gas": "#A66A3F",
    "Biomass": "#6FA85C",
    "Nuclear": "#C44E52",
    "Other": "#BDBDBD",
    "Geothermal": "#8C564B",
    "Hydro": "#4C78A8",
    "Solar": "#F2C14E",
    "Wind": "#72B7B2",
    "Storage": "mediumorchid",
}


def _to_dataframe(iamdf) -> pd.DataFrame:
    if hasattr(iamdf, "as_pandas"):
        return iamdf.as_pandas()

    if hasattr(iamdf, "data"):
        return iamdf.data.copy()

    if isinstance(iamdf, pd.DataFrame):
        return iamdf.copy()

    raise TypeError("Expected pyam.IamDataFrame or pandas.DataFrame-like object.")


def _prepare_power_capacity_data(
    iamdf,
    year: int,
    region: str,
    scenario: str,
) -> pd.DataFrame:
    d = iamdf.filter(
        variable=POWER_CAPACITY_VARIABLES,
        region=region,
        scenario=scenario,
        year=year,
    )

    df = _to_dataframe(d)

    if df.empty:
        raise ValueError(
            f"No installed-capacity data found for "
            f"region={region}, scenario={scenario}, year={year}."
        )

    wide = (
        df.pivot_table(
            index="model",
            columns="variable",
            values="value",
            aggfunc="sum",
        )
        .reindex(columns=POWER_CAPACITY_VARIABLES)
        .fillna(0.0)
        .sort_index()
    )

    return wide


def _get_base_label(label: str) -> str:
    return label.replace(" w/ CCS", "").replace(" w/o CCS", "")


def _get_stack_variables() -> list[str]:
    return [
        v for v in POWER_CAPACITY_VARIABLES
        if "Storage|Power" not in v
    ]


def _normalise_to_share(
    wide: pd.DataFrame,
    stack_vars: list[str],
    storage_var: str,
) -> pd.DataFrame:
    """
    Convert GW values to percentage shares.

    Generation technologies are normalised by total generation capacity.
    Storage lollipop is shown as storage power capacity divided by total
    generation capacity, i.e. storage is not included in the denominator.
    """
    out = wide.copy()

    stack_total = out[stack_vars].sum(axis=1).replace(0, np.nan)

    out[stack_vars] = out[stack_vars].div(stack_total, axis=0) * 100.0

    if storage_var in out.columns:
        out[storage_var] = out[storage_var].div(stack_total, axis=0) * 100.0

    return out.fillna(0.0)


def _make_legend_handles(stack_vars: list[str]) -> list[Patch | Line2D]:
    handles: list[Patch | Line2D] = []

    for var in stack_vars:
        label = POWER_CAPACITY_LABELS[var]
        base_label = _get_base_label(label)
        color = POWER_CAPACITY_COLORS.get(base_label, "#BDBDBD")
        hatch = "///" if "w/ CCS" in label else None

        handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                linewidth=0.35,
                hatch=hatch,
                label=label,
            )
        )

    handles.append(
        Line2D(
            [0],
            [0],
            color=POWER_CAPACITY_COLORS["Storage"],
            marker="o",
            markerfacecolor=POWER_CAPACITY_COLORS["Storage"],
            markeredgecolor="black",
            linewidth=1.2,
            markersize=6,
            label="Storage power capacity",
        )
    )

    return handles


def _plot_one_capacity_panel(
    ax,
    wide: pd.DataFrame,
    year: int,
    storage_var: str,
    as_share: bool,
):
    stack_vars = _get_stack_variables()

    if as_share:
        plot_wide = _normalise_to_share(
            wide=wide,
            stack_vars=stack_vars,
            storage_var=storage_var,
        )
    else:
        plot_wide = wide.copy()

    models = plot_wide.index.astype(str).tolist()
    x = np.arange(len(models))

    bottom = np.zeros(len(models))

    for var in stack_vars:
        label = POWER_CAPACITY_LABELS[var]
        base_label = _get_base_label(label)

        color = POWER_CAPACITY_COLORS.get(base_label, "#BDBDBD")
        hatch = "///" if "w/ CCS" in label else None

        values = plot_wide[var].to_numpy(dtype=float)

        ax.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            hatch=hatch,
        )

        bottom += values
    if not as_share:
        storage_values = plot_wide[storage_var].to_numpy(dtype=float)
    
        ax.vlines(
            x,
            bottom,
            bottom + storage_values,
            color=POWER_CAPACITY_COLORS["Storage"],
            linewidth=1.2,
            alpha=0.95,
            zorder=4,
        )
    
        ax.scatter(
            x,
            bottom + storage_values,
            s=42,
            marker="o",
            facecolor=POWER_CAPACITY_COLORS["Storage"],
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

    ax.set_title(str(year), fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)

    ax.grid(axis="y", linewidth=0.45, alpha=0.20)
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _plot_capacity_stack_two_years(
    wide_by_year: dict[int, pd.DataFrame],
    years: tuple[int, int],
    region: str,
    scenario: str,
    storage_var: str,
    as_share: bool,
    outfile_stem: str | None,
    save_png: bool,
    save_pdf: bool,
    save_svg: bool,
):
    stack_vars = _get_stack_variables()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for ax, year in zip(axes, years):
        _plot_one_capacity_panel(
            ax=ax,
            wide=wide_by_year[year],
            year=year,
            storage_var=storage_var,
            as_share=as_share,
        )

    ylabel = (
        "Share of installed generation capacity (%)"
        if as_share
        else "Installed power capacity (GW)"
    )
    axes[0].set_ylabel(ylabel)

    title_suffix = "technology shares" if as_share else "installed capacity"
    fig.suptitle(
        f"Installed generation {title_suffix} and storage power capacity\n"
        f"{region.replace('(*)', '').strip()}, {scenario}",
        fontsize=11,
        y=1.0,
    )

    legend_handles = _make_legend_handles(stack_vars)

    axes[-1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8.2,
        title="Technology",
        title_fontsize=9,
    )

    fig.subplots_adjust(
        right=0.82,
        bottom=0.22,
        top=0.82,
    )

    if as_share:
        axes[0].set_ylim(0, 100)

    if outfile_stem:
        save_figure(
            fig,
            ANALYSIS_OUTPUT_DIR / outfile_stem,
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, axes


def make_stacked_bar(
    years: tuple[int, int] = (2030, 2050),
    region: str = SELECTED_REGION,
    scenario: str = SELECTED_SCENARIO,
    as_share: bool = False,
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
):
    """
    New stacked-bar diagnostic figure.

    Generation technologies are shown as stacked bars for two years side by side.
    Electricity storage power capacity is shown as an unscaled lollipop marker
    above each model-specific generation-capacity stack.

    Parameters
    ----------
    years
        Two years shown as a 1 x 2 subplot.
    as_share
        If False, show values in GW.
        If True, show generation technologies as percentage shares of total
        generation capacity. Storage is shown as storage power capacity divided
        by total generation capacity.
    """
    if len(years) != 2:
        raise ValueError("'years' must contain exactly two years, e.g. (2030, 2050).")

    _, iamdf, storage_var = load_analysis_data()

    wide_by_year = {
        year: _prepare_power_capacity_data(
            iamdf=iamdf,
            year=year,
            region=region,
            scenario=scenario,
        )
        for year in years
    }

    suffix = "share" if as_share else "gw"
    outfile_stem = (
        f"stacked_bar_power_capacity_{scenario}_{years[0]}_{years[1]}_{suffix}"
        .replace("|", "_")
    )

    fig, _ = _plot_capacity_stack_two_years(
        wide_by_year=wide_by_year,
        years=years,
        region=region,
        scenario=scenario,
        storage_var=storage_var,
        as_share=as_share,
        outfile_stem=outfile_stem,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    source_data = []
    for year, wide in wide_by_year.items():
        tmp = wide.reset_index().rename(columns=POWER_CAPACITY_LABELS)
        tmp.insert(1, "year", year)
        source_data.append(tmp)

    source_data = pd.concat(source_data, ignore_index=True)

    return fig, source_data