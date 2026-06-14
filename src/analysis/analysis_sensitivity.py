from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.common.config import (
    SCENARIO_COLORS,
    ANALYSIS_OUTPUT_DIR,
    SELECTED_REGION,
    FIG4_SCEN_NZ,
    VAR_STORAGE_REL_PEAK,
)
from src.common.plot_utils import save_figure, horizontal_grid_only
from src.analysis.scatter_plots import (
    _prepare_base_df,
    _add_peak_load_and_storage_ratio,
)
from src.analysis.data_loader import load_analysis_data, build_model_markers


DEFAULT_SCENARIO_SET = [
    "NetZero",
    "NetZero|LimBio",
    "NetZero|LimCCS",
    "NetZero|LimNuc",
]

DEFAULT_EXCLUDE_MODELS = [
    "MEESA",
    "GCAM",
    "TIAM-ECN",
]


def prepare_storage_sensitivity_data(
    year: int = 2050,
    selected_region: str = SELECTED_REGION,
    reference_scenario: str = FIG4_SCEN_NZ,
    scenario_set: list[str] | tuple[str, ...] | None = None,
    exclude_models: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare storage-intensity levels and scenario deltas for within-model
    sensitivity analysis.

    Returns
    -------
    df_storage : pd.DataFrame
        Storage intensity by model, scenario, and year.
    df_delta : pd.DataFrame
        Storage intensity with reference values, absolute deltas, and relative
        changes against the reference scenario.
    diagnostic_scenarios : list[str]
        Scenario names excluding the reference scenario.
    """
    if scenario_set is None:
        scenario_set = DEFAULT_SCENARIO_SET
    scenario_set = list(scenario_set)

    if exclude_models is None:
        exclude_models = DEFAULT_EXCLUDE_MODELS
    exclude_models = list(exclude_models)

    diagnostic_scenarios = [s for s in scenario_set if s != reference_scenario]

    _, eu_nzero, _ = load_analysis_data()
    iam_models = [m for m in eu_nzero.model if m not in exclude_models]

    df_iam = _prepare_base_df(
        selected_region=selected_region,
        years=[year],
        scenario_set=scenario_set,
    )
    df_iam = df_iam.filter(model=iam_models)
    df_iam = _add_peak_load_and_storage_ratio(df_iam)

    ts = (
        df_iam
        .filter(variable=VAR_STORAGE_REL_PEAK, year=year)
        .timeseries()
        .reset_index()
    )

    df_storage = ts[["model", "scenario", "region", "variable", year]].copy()
    df_storage = df_storage.rename(columns={year: "storage_intensity"})
    df_storage["year"] = year
    df_storage["storage_intensity"] = pd.to_numeric(
        df_storage["storage_intensity"], errors="coerce"
    )
    df_storage = df_storage.dropna(subset=["storage_intensity"])

    models_with_ref = set(
        df_storage.loc[
            df_storage["scenario"] == reference_scenario,
            "model",
        ].unique()
    )
    df_storage = df_storage[df_storage["model"].isin(models_with_ref)].copy()

    ref_df = (
        df_storage[df_storage["scenario"] == reference_scenario]
        [["model", "storage_intensity"]]
        .rename(columns={"storage_intensity": "ref_storage_intensity"})
    )

    df_delta = pd.merge(df_storage, ref_df, on="model", how="inner")
    df_delta["delta_storage_intensity"] = (
        df_delta["storage_intensity"] - df_delta["ref_storage_intensity"]
    )
    df_delta["relative_change_pct"] = np.where(
        df_delta["ref_storage_intensity"].abs() > 1e-9,
        df_delta["delta_storage_intensity"]
        / df_delta["ref_storage_intensity"]
        * 100,
        np.nan,
    )

    return df_storage, df_delta, diagnostic_scenarios


def build_sensitivity_table(
    df_storage: pd.DataFrame,
    df_delta: pd.DataFrame,
    reference_scenario: str = FIG4_SCEN_NZ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the scenario-delta table and compact within-model sensitivity table.
    """
    df_delta_diag = df_delta[df_delta["scenario"] != reference_scenario].copy()

    spread_all = (
        df_storage
        .groupby("model")
        .agg(
            n_scenarios=("scenario", "nunique"),
            min_storage_intensity=("storage_intensity", "min"),
            max_storage_intensity=("storage_intensity", "max"),
            mean_storage_intensity=("storage_intensity", "mean"),
            median_storage_intensity=("storage_intensity", "median"),
            sd_storage_intensity=("storage_intensity", "std"),
        )
        .reset_index()
    )

    spread_all["scenario_range_storage_intensity"] = (
        spread_all["max_storage_intensity"] - spread_all["min_storage_intensity"]
    )
    spread_all["scenario_cv_storage_intensity"] = (
        spread_all["sd_storage_intensity"] / spread_all["mean_storage_intensity"]
    )
    spread_all["max_min_ratio_storage_intensity"] = np.where(
        spread_all["min_storage_intensity"] > 0,
        spread_all["max_storage_intensity"] / spread_all["min_storage_intensity"],
        np.nan,
    )

    if df_delta_diag.empty:
        spread_delta = pd.DataFrame({"model": spread_all["model"]})
        most_sensitive = pd.DataFrame({"model": spread_all["model"]})
    else:
        spread_delta = (
            df_delta_diag
            .groupby("model")
            .agg(
                min_delta_storage_intensity=("delta_storage_intensity", "min"),
                max_delta_storage_intensity=("delta_storage_intensity", "max"),
                mean_abs_delta_storage_intensity=(
                    "delta_storage_intensity",
                    lambda x: x.abs().mean(),
                ),
                max_abs_delta_storage_intensity=(
                    "delta_storage_intensity",
                    lambda x: x.abs().max(),
                ),
                mean_abs_relative_change_pct=(
                    "relative_change_pct",
                    lambda x: x.abs().mean(),
                ),
                max_abs_relative_change_pct=(
                    "relative_change_pct",
                    lambda x: x.abs().max(),
                ),
            )
            .reset_index()
        )

        idx = (
            df_delta_diag
            .groupby("model")["delta_storage_intensity"]
            .apply(lambda x: x.abs().idxmax())
        )

        most_sensitive = (
            df_delta_diag
            .loc[
                idx,
                [
                    "model",
                    "scenario",
                    "delta_storage_intensity",
                    "relative_change_pct",
                ],
            ]
            .rename(
                columns={
                    "scenario": "most_sensitive_scenario",
                    "delta_storage_intensity": "most_sensitive_delta_storage_intensity",
                    "relative_change_pct": "most_sensitive_relative_change_pct",
                }
            )
        )

    sensitivity_table = (
        spread_all
        .merge(spread_delta, on="model", how="left")
        .merge(most_sensitive, on="model", how="left")
        .sort_values("model")
        .reset_index(drop=True)
    )

    # rounded export-friendly copy
    for c in sensitivity_table.select_dtypes(include=["number"]).columns:
        sensitivity_table[c] = sensitivity_table[c].round(4)
    for c in df_delta_diag.select_dtypes(include=["number"]).columns:
        df_delta_diag[c] = df_delta_diag[c].round(4)

    return df_delta_diag.sort_values(["model", "scenario"]).reset_index(drop=True), sensitivity_table


def plot_storage_scenario_sensitivity(
    df_delta: pd.DataFrame,
    diagnostic_scenarios: list[str],
    scenario_set: list[str] | tuple[str, ...] | None = None,
    reference_scenario: str = FIG4_SCEN_NZ,
    year: int = 2050,
    plot_mode: str = "range",
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    outfile_stem: str | None = None,
    row_spacing: float = 0.72,
    range_linewidth: float = 4.0,
    dot_size: float = 62,
):
    """
    Plot within-model storage sensitivity as either:
    - plot_mode='range': actual storage-intensity range across scenarios
    - plot_mode='delta': scenario differences relative to the reference scenario
    """
    if scenario_set is None:
        scenario_set = DEFAULT_SCENARIO_SET
    scenario_set = list(scenario_set)

    if plot_mode not in {"range", "delta"}:
        raise ValueError("plot_mode must be either 'delta' or 'range'.")

    _, eu_nzero, _ = load_analysis_data()
    model_markers = build_model_markers(eu_nzero)

    df_delta_diag = df_delta[df_delta["scenario"] != reference_scenario].copy()
    model_order = sorted(df_delta_diag["model"].unique())

    if not model_order:
        raise ValueError("No diagnostic scenario observations available for plotting.")

    y_positions = {model: i * row_spacing for i, model in enumerate(model_order)}

    fig_height = max(2.8, 0.45 * len(model_order))
    fig, ax = plt.subplots(figsize=(7.0, fig_height))

    if plot_mode == "delta":
        range_df = (
            df_delta_diag
            .groupby("model")
            .agg(
                min_x=("delta_storage_intensity", "min"),
                max_x=("delta_storage_intensity", "max"),
            )
            .reset_index()
        )

        for _, row in range_df.iterrows():
            ax.hlines(
                y=y_positions[row["model"]],
                xmin=row["min_x"],
                xmax=row["max_x"],
                color="0.70",
                linewidth=range_linewidth,
                zorder=1,
            )

        for scen in diagnostic_scenarios:
            g = df_delta_diag[df_delta_diag["scenario"] == scen].copy()
            for _, row in g.iterrows():
                model = str(row["model"])
                ax.scatter(
                    row["delta_storage_intensity"],
                    y_positions[model],
                    s=dot_size,
                    marker=model_markers.get(model, "o"),
                    color=SCENARIO_COLORS.get(scen, "grey"),
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=3,
                )

        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)
        xlabel = (
            f"Change in electricity storage intensity relative to {reference_scenario}, {year}\n"
            "(GW storage per GW peak demand)"
        )
        legend_title = "Scenario perturbation"
        legend_scenarios = diagnostic_scenarios

    else:
        df_range_plot = df_delta.copy()
        df_range_plot["model"] = pd.Categorical(
            df_range_plot["model"],
            categories=model_order,
            ordered=True,
        )

        range_df = (
            df_range_plot
            .groupby("model", observed=True)
            .agg(
                min_x=("storage_intensity", "min"),
                max_x=("storage_intensity", "max"),
            )
            .reset_index()
        )

        for _, row in range_df.iterrows():
            ax.hlines(
                y=y_positions[str(row["model"])],
                xmin=row["min_x"],
                xmax=row["max_x"],
                color="0.70",
                linewidth=range_linewidth,
                zorder=1,
            )

        for scen in scenario_set:
            g = df_range_plot[df_range_plot["scenario"] == scen].copy()
            for _, row in g.iterrows():
                model = str(row["model"])
                is_ref = scen == reference_scenario
                ax.scatter(
                    row["storage_intensity"],
                    y_positions[model],
                    s=dot_size + 16 if is_ref else dot_size,
                    marker=model_markers.get(model, "o"),
                    color=SCENARIO_COLORS.get(scen, "grey"),
                    edgecolor="black",
                    linewidth=1.2 if is_ref else 0.7,
                    zorder=4 if is_ref else 3,
                )

        xlabel = (
            f"Electricity storage deployment across scenarios, {year}\n"
            "(GW power capacity per GW peak demand)"
        )
        legend_title = "Scenario"
        legend_scenarios = scenario_set

    ax.set_yticks([y_positions[m] for m in model_order])
    ax.set_yticklabels(model_order)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Model")

    ax.set_title(
        "Within-model scenario sensitivity of electricity storage deployment",
        loc="left",
        x=-0.1,
        fontweight="bold",
        fontsize=10,
        color="0.15",
    )

    horizontal_grid_only(ax, linewidth=0.7, alpha=0.25)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=4, width=0.8, color="black", direction="out")

    y_min = min(y_positions.values()) - row_spacing * 0.7
    y_max = max(y_positions.values()) + row_spacing * 0.7
    ax.set_ylim(y_max, y_min)

    legend_marker = "|"
    legend_handles = [
        Line2D(
            [0], [0],
            marker=legend_marker,
            linestyle="",
            label=scen.replace("NetZero|", ""),
            color=SCENARIO_COLORS.get(scen, "grey"),
            markeredgecolor=SCENARIO_COLORS.get(scen, "grey"),
            markerfacecolor=SCENARIO_COLORS.get(scen, "grey"),
            markeredgewidth=3.8,
            markersize=12,
        )
        for scen in legend_scenarios
    ]

    ax.legend(
        handles=legend_handles,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    plt.tight_layout()

    if outfile_stem is None:
        outfile_stem = f"analysis2_storage_scenario_sensitivity_{year}_{plot_mode}"

    if save_png or save_pdf or save_svg:
        save_figure(
            fig,
            ANALYSIS_OUTPUT_DIR / outfile_stem,
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, ax


def make_analysis2_sensitivity(
    year: int = 2050,
    selected_region: str = SELECTED_REGION,
    reference_scenario: str = FIG4_SCEN_NZ,
    scenario_set: list[str] | tuple[str, ...] | None = None,
    exclude_models: list[str] | tuple[str, ...] | None = None,
    plot_mode: str = "range",
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    save_csv: bool = True,
):
    """
    Main workflow entry point for Analysis 2.

    Computes within-model scenario sensitivity of electricity storage intensity,
    saves CSV tables, and creates the scenario-sensitivity dot-range plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    outputs : dict[str, pd.DataFrame]
        Contains storage levels, scenario deltas, diagnostic deltas, and the
        model-level sensitivity table.
    """
    if scenario_set is None:
        scenario_set = DEFAULT_SCENARIO_SET
    scenario_set = list(scenario_set)

    df_storage, df_delta, diagnostic_scenarios = prepare_storage_sensitivity_data(
        year=year,
        selected_region=selected_region,
        reference_scenario=reference_scenario,
        scenario_set=scenario_set,
        exclude_models=exclude_models,
    )

    df_delta_diag, sensitivity_table = build_sensitivity_table(
        df_storage=df_storage,
        df_delta=df_delta,
        reference_scenario=reference_scenario,
    )

    outfile_stem = f"analysis2_storage_scenario_sensitivity_{year}_{plot_mode}"

    fig, _ = plot_storage_scenario_sensitivity(
        df_delta=df_delta,
        diagnostic_scenarios=diagnostic_scenarios,
        scenario_set=scenario_set,
        reference_scenario=reference_scenario,
        year=year,
        plot_mode=plot_mode,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
        outfile_stem=outfile_stem,
    )

    if save_csv:
        ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_storage.to_csv(
            ANALYSIS_OUTPUT_DIR / f"analysis2_storage_intensity_{year}.csv",
            index=False,
        )
        df_delta.to_csv(
            ANALYSIS_OUTPUT_DIR / f"analysis2_storage_scenario_deltas_{year}.csv",
            index=False,
        )
        df_delta_diag.to_csv(
            ANALYSIS_OUTPUT_DIR / f"analysis2_storage_diagnostic_deltas_{year}.csv",
            index=False,
        )
        sensitivity_table.to_csv(
            ANALYSIS_OUTPUT_DIR / f"analysis2_sensitivity_table_{year}.csv",
            index=False,
        )

    outputs = {
        "storage_intensity": df_storage,
        "scenario_deltas": df_delta,
        "diagnostic_deltas": df_delta_diag,
        "sensitivity_table": sensitivity_table,
    }

    return fig, outputs


if __name__ == "__main__":
    fig, outputs = make_analysis2_sensitivity()
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)

    print("\n=== Analysis 2: scenario deltas relative to NetZero ===")
    print(outputs["diagnostic_deltas"])

    print("\n=== Analysis 2: within-model scenario-sensitivity table ===")
    print(outputs["sensitivity_table"])

    plt.show()
