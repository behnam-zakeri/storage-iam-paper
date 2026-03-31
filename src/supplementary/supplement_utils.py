from __future__ import annotations

import pandas as pd
from pyam import IamDataFrame
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.common.config import (
    SELECTED_REGION,
    SCENARIOS_DIAGNOSTIC,
    SCENARIO_COLORS,
    VAR_CAP_WIND,
    VAR_CAP_SOLAR,
    VAR_CAP_VRE,
    VAR_STORAGE_REL_VRE,
    VAR_ELEC_WIND,
    VAR_ELEC_SOLAR,
    VAR_ELEC_VRE,
    VAR_ELEC_TOTAL,
    VAR_VRE_SHARE,
    VAR_ELEC_CAPACITY,
    VAR_STORAGE_REL_TOTAL,
)
from src.analysis.data_loader import load_analysis_data, build_model_markers


def get_default_supplement_subset(
    selected_region: str = SELECTED_REGION,
    scenarios: list[str] | None = None,
    years=None,
) -> tuple[IamDataFrame, str]:
    """
    Return the default Europe/scenario subset for supplementary figures.
    """
    if scenarios is None:
        scenarios = SCENARIOS_DIAGNOSTIC

    _, eu_nzero, storage_var = load_analysis_data()

    df = eu_nzero.filter(
        region=selected_region,
        scenario=scenarios,
        year=years,
    )
    return df, storage_var


def get_storage_reporting_subset(
    selected_region: str = SELECTED_REGION,
    scenarios: list[str] | None = None,
    years=None,
) -> tuple[IamDataFrame, IamDataFrame, str]:
    """
    Return:
    - filtered net-zero subset
    - subset of models reporting storage
    - storage variable name
    """
    df, storage_var = get_default_supplement_subset(
        selected_region=selected_region,
        scenarios=scenarios,
        years=years,
    )

    eu_storage = df.filter(model=df.filter(variable=storage_var).model)
    return df, eu_storage, storage_var


def add_vre_capacity(df: IamDataFrame) -> IamDataFrame:
    """
    Add VRE capacity = wind + solar capacity.
    """
    out = df.copy()
    out.add(
        VAR_CAP_WIND,
        VAR_CAP_SOLAR,
        VAR_CAP_VRE,
        append=True,
    )
    return out


def add_storage_to_vre_ratio(df: IamDataFrame, storage_var: str) -> IamDataFrame:
    """
    Add storage power / VRE installed capacity.
    """
    out = add_vre_capacity(df)
    out.divide(
        storage_var,
        VAR_CAP_VRE,
        VAR_STORAGE_REL_VRE,
        append=True,
    )
    return out


def compute_relative_change_vs_reference(
    df: pd.DataFrame,
    *,
    target_variable: str,
    reference_scenario: str,
    year: int,
    exclude_models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute relative change (%) of a target variable against a reference scenario
    for each model.
    """
    out = df.copy()

    if exclude_models:
        out = out[~out["model"].isin(exclude_models)].copy()

    out = out[out["variable"] == target_variable].copy()

    ref_df = out[out["scenario"] == reference_scenario].copy()
    ref_df = ref_df.rename(columns={year: "ref_value"})
    ref_df = ref_df[["model", "ref_value"]]

    merged = pd.merge(out, ref_df, on="model", how="inner")
    merged = merged[merged["scenario"] != reference_scenario].copy()

    merged["relative_change"] = (
        (merged[year] - merged["ref_value"]) / merged["ref_value"] * 100.0
    )

    merged = merged.sort_values(["model", "scenario"]).reset_index(drop=True)
    return merged


def add_vre_generation_and_share(df: IamDataFrame) -> IamDataFrame:
    """
    Add VRE generation and VRE share in total electricity generation.
    """
    out = df.copy()
    out.add(
        VAR_ELEC_WIND,
        VAR_ELEC_SOLAR,
        VAR_ELEC_VRE,
        append=True,
    )
    out.divide(
        VAR_ELEC_VRE,
        VAR_ELEC_TOTAL,
        VAR_VRE_SHARE,
        append=True,
    )
    return out


def add_storage_to_total_ratio(df: IamDataFrame, storage_var: str) -> IamDataFrame:
    """
    Add storage power relative to total electricity capacity.
    """
    out = df.copy()
    out.divide(
        storage_var,
        VAR_ELEC_CAPACITY,
        VAR_STORAGE_REL_TOTAL,
        append=True,
    )
    return out


def add_flexible_capacity_share(
    df: IamDataFrame,
    flexible_var: str = "Capacity|Electricity|Flexible",
    flexible_share_var: str = "Relative Capacity|Electricity|Flexible to Total",
) -> IamDataFrame:
    """
    Add flexible generation capacity (gas + hydro) and its share in total capacity.
    """
    out = df.copy()
    out.add(
        "Capacity|Electricity|Gas",
        "Capacity|Electricity|Hydro",
        flexible_var,
        append=True,
    )
    out.divide(
        flexible_var,
        VAR_ELEC_CAPACITY,
        flexible_share_var,
        append=True,
    )
    return out


def filter_out_scenarios(
    df: IamDataFrame,
    exclude_patterns: list[str] | None = None,
) -> IamDataFrame:
    """
    Exclude scenarios matching wildcard patterns.
    """
    out = df
    if exclude_patterns:
        out = out.filter(scenario=exclude_patterns, keep=False)
    return out


def build_yearly_scatter_table(
    df: IamDataFrame,
    var_x: str,
    var_y: str,
    years: list[int],
) -> pd.DataFrame:
    """
    Build a long plotting table for supplementary scatter plots.
    One row per (model, scenario, year) with columns var_x and var_y.
    """
    df_sub = df.filter(variable=[var_x, var_y])

    tables = []
    ts = df_sub.timeseries().reset_index()

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


def plot_supplement_scatter_by_year(
    df_plot: pd.DataFrame,
    *,
    var_x: str,
    var_y: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str,
    marker_size: float = 100,
    legend_frameon: bool = False,
):
    """
    Supplementary scatter style:
    - no connecting lines
    - model-specific markers
    - scenario colours
    - simple legend with model markers + scenario patches
    """
    _, eu_nzero, _ = load_analysis_data()
    model_markers = build_model_markers(eu_nzero)

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df_plot.iterrows():
        x = row[var_x]
        y = row[var_y]
        model = row["model"]
        scenario = row["scenario"]

        ax.scatter(
            x,
            y,
            marker=model_markers.get(model, "o"),
            color=SCENARIO_COLORS[scenario],
            s=marker_size,
            edgecolor="grey",
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    legend_elements = []

    for model in sorted(df_plot["model"].dropna().unique()):
        legend_elements.append(
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
        )

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

