from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.common.config import (
    ANALYSIS_OUTPUT_DIR,
    PYPSA_XLSX,
    PYPSA_SHEET,
    ENTSOE_POLICY_DATA,
    EU_INSTALLED_2025_STORAGE_GW,
    FIG4_YEARS,
    FIG4_SCEN_NZ,
    SCENARIOS_DIAGNOSTIC,
    SCENARIO_COLORS,
    SELECTED_REGION,
    VAR_STORAGE_POWER,
)
from src.common.plot_utils import save_figure
from src.analysis.data_loader import load_analysis_data, get_pypsa_storage_row_label


def _read_pypsa_storage_distribution() -> np.ndarray:
    """
    Read the PyPSA storage row from the benchmark workbook and return the
    full distribution across weather years as a 1D numpy array.
    """
    pypsa_df = pd.read_excel(PYPSA_XLSX, sheet_name=PYPSA_SHEET)
    pypsa_df["tech"] = pypsa_df["tech"].astype(str)

    storage_row_label = get_pypsa_storage_row_label()

    row = pypsa_df.loc[pypsa_df["tech"].str.strip() == storage_row_label]
    if row.empty:
        raise ValueError(
            f"Row '{storage_row_label}' not found in sheet '{PYPSA_SHEET}'."
        )

    row = row.iloc[0]
    year_cols = [c for c in pypsa_df.columns if c != "tech"]
    pypsa_values = pd.to_numeric(row[year_cols], errors="coerce").dropna().to_numpy()

    return pypsa_values


def _build_policy_data(
    eu_nzero,
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    storage_var: str = VAR_STORAGE_POWER,
) -> dict[str, dict[int, float]]:
    """
    Build benchmark dictionary for ENTSO-E and MEESA.
    """
    if years is None:
        years = FIG4_YEARS

    policy_data = {"ENTSO-E": dict(ENTSOE_POLICY_DATA), "MEESA": {}}

    df_meesa = eu_nzero.filter(
        region=selected_region,
        scenario=FIG4_SCEN_NZ,
        year=years,
        variable=storage_var,
        model="MEESA",
    )

    if not df_meesa.empty:
        tmp = df_meesa.timeseries().reset_index().set_index("model")
        for model_name, yr in product(tmp.index, years):
            if yr in tmp.columns:
                policy_data[model_name][yr] = tmp.loc[model_name, yr]

    return policy_data


def _prepare_boxplot_table(
    eu_nzero,
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    scenarios: list[str] | None = None,
    storage_var: str = VAR_STORAGE_POWER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare long-format IAM table for Figure 4a.

    Returns
    -------
    df_all : DataFrame
        All models including benchmarks such as MEESA.
    df_iam : DataFrame
        IAM-only table excluding MEESA from boxplots.
    """
    if years is None:
        years = FIG4_YEARS
    if scenarios is None:
        scenarios = SCENARIOS_DIAGNOSTIC

    df = eu_nzero.filter(
        region=selected_region,
        scenario=scenarios,
        year=years,
        variable=storage_var,
    )

    df = df.timeseries().reset_index()
    df = df.melt(
        id_vars=["model", "scenario", "region", "variable", "unit"],
        var_name="year",
        value_name="value",
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["year", "value"], inplace=True)

    df_iam = df.loc[df["model"] != "MEESA"].copy()

    return df, df_iam


def _build_legend_handles(
    hue_order: list[str],
    pypsa_hatch: str = "////",
) -> list:
    """
    Build the custom legend used in Figure 4a.
    """
    legend_elements = []

    # Scenarios
    legend_elements.append(Line2D([], [], linestyle="none", label="Scenarios"))
    for scen in hue_order:
        legend_elements.append(
            Patch(facecolor=SCENARIO_COLORS[scen], edgecolor="none", label=scen)
        )

    # spacer
    legend_elements.append(Line2D([], [], linestyle="none", label=""))

    # Benchmarks
    legend_elements.append(Line2D([], [], linestyle="none", label="Benchmarks"))
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="grey",
            marker="D",
            linestyle="--",
            markerfacecolor="white",
            markeredgecolor="black",
            label="ENTSO-E (TYNDP 2024)",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="grey",
            marker="^",
            linestyle="--",
            markerfacecolor="white",
            markeredgecolor="black",
            label="MEESA (NetZero)",
        )
    )
    legend_elements.append(
        Patch(
            facecolor="none",
            edgecolor="black",
            hatch=pypsa_hatch,
            label="PyPSA-Eur (60 weather years)",
        )
    )

    return legend_elements


def make_figure4a(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    offset_mean: bool = False,
    selected_region: str = SELECTED_REGION,
    scenarios: list[str] | None = None,
    years: list[int] | None = None,
):
    """
    Manuscript Figure 4a:
    Electricity storage in IAM net-zero scenarios vs benchmark studies.

    Returns
    -------
    fig : matplotlib Figure
    source_data : DataFrame
        Tidy source-data table including IAM values, benchmark points,
        and PyPSA weather-year distribution.
    """
    if years is None:
        years = FIG4_YEARS
    if scenarios is None:
        scenarios = SCENARIOS_DIAGNOSTIC

    _, eu_nzero, storage_var = load_analysis_data()

    policy_data = _build_policy_data(
        eu_nzero=eu_nzero,
        selected_region=selected_region,
        years=years,
        storage_var=storage_var,
    )

    pypsa_values = _read_pypsa_storage_distribution()
    df_all, df_iam = _prepare_boxplot_table(
        eu_nzero=eu_nzero,
        selected_region=selected_region,
        years=years,
        scenarios=scenarios,
        storage_var=storage_var,
    )

    hue_order = list(scenarios)
    year_positions = {year: i for i, year in enumerate(sorted(years))}
    supply_offsets = {
        scen: (i - (len(hue_order) - 1) / 2) * 0.15 for i, scen in enumerate(hue_order)
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.boxplot(
        data=df_iam,
        x="year",
        y="value",
        hue="scenario",
        hue_order=hue_order,
        palette=SCENARIO_COLORS,
        width=0.3,
        fliersize=4,
        ax=ax,
    )

    # Right-of-cluster x positions for benchmarks
    box_offsets = supply_offsets.copy()
    cluster_right = max(box_offsets.values()) + 0.02
    meesa_xs = [year_positions[y] + cluster_right for y in sorted(years)]
    entsoe_xs = [x + 0.03 for x in meesa_xs]

    # ENTSO-E points
    entsoe_ys = [ENTSOE_POLICY_DATA[y] for y in sorted(years)]

    # MEESA benchmark: NetZero only
    meesa_nz = (
        df_all[(df_all["model"] == "MEESA") & (df_all["scenario"] == FIG4_SCEN_NZ)]
        .groupby("year", as_index=False)["value"]
        .mean()
    )
    meesa_map = dict(zip(meesa_nz["year"], meesa_nz["value"]))
    meesa_ys = [meesa_map[y] for y in sorted(years)]

    ax.plot(
        meesa_xs,
        meesa_ys,
        color="grey",
        linewidth=0,
        alpha=0.8,
        marker="^",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        label="MEESA (NetZero)",
    )

    ax.plot(
        entsoe_xs,
        entsoe_ys,
        color="grey",
        linewidth=0,
        alpha=0.8,
        marker="D",
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor="black",
        label="ENTSO-E (TYNDP 2024)",
    )

    # Means / medians
    if not offset_mean:
        supply_offsets = {
            scen: (i - (len(hue_order) - 1) / 2) * 0.075
            for i, scen in enumerate(hue_order)
        }

    means = df_iam.groupby(["year", "scenario"], as_index=False)["value"].mean()
    means["x"] = means["year"].map(year_positions) + means["scenario"].map(supply_offsets)

    netzero_medians = (
        df_iam[df_iam["scenario"] == FIG4_SCEN_NZ]
        .groupby("year", as_index=False)["value"]
        .median()
    )
    netzero_medians["x"] = (
        netzero_medians["year"].map(year_positions) + supply_offsets[FIG4_SCEN_NZ]
    )

    for _, row in netzero_medians.iterrows():
        ax.text(
            row["x"] - 0.06,
            row["value"],
            f"{row['value']:.0f}",
            ha="right",
            va="center",
            fontsize=10,
            color="grey",
            zorder=7,
        )

    # PyPSA box slightly right of 2050 cluster
    pypsa_x = year_positions[2050] + 0.25

    bp = ax.boxplot(
        [pypsa_values],
        positions=[pypsa_x],
        widths=0.08,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
    )

    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor="black", linewidth=1.3, hatch="////")
    for k in ["whiskers", "caps", "medians"]:
        for artist in bp[k]:
            artist.set(color="black", linewidth=1.3)

    ax.text(
        pypsa_x + 0.07,
        np.median(pypsa_values),
        f"{np.median(pypsa_values):.0f}",
        ha="left",
        va="center",
        fontsize=10,
        color="grey",
        zorder=8,
    )

    # Installed capacity reference line
    ax.axhline(
        EU_INSTALLED_2025_STORAGE_GW,
        color="grey",
        linewidth=0.8,
        linestyle="--",
        alpha=0.4,
    )
    ax.text(
        pypsa_x - 0.7,
        EU_INSTALLED_2025_STORAGE_GW + 5,
        f"Installed by end 2025: ~{EU_INSTALLED_2025_STORAGE_GW} GW",
        fontsize=10,
        color="grey",
        zorder=8,
    )

    # Custom legend
    legend_elements = _build_legend_handles(hue_order=hue_order)
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        handlelength=1.2,
        handletextpad=0.6,
        labelspacing=0.5,
    )

    # Final styling
    base_ticks = [year_positions[y] for y in sorted(years)]
    labels = [str(y) for y in sorted(years)]

    ax.set_xticks(base_ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Electricity storage power capacity (GW)", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_title(
        "a) Electricity storage in IAM net-zero scenarios (EU27 & UK) vs. benchmark studies",
        fontweight="bold",
    )
    ax.set_ylim(0, df_all["value"].max() + 100)

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    if save_png or save_pdf or save_svg:
        save_figure(
            fig,
            ANALYSIS_OUTPUT_DIR / "Figure-4a",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    # -----------------------------------------------------
    # Source data export table
    # -----------------------------------------------------
    source_iam = df_all.copy()
    source_iam["source_type"] = np.where(
        source_iam["model"] == "MEESA", "benchmark_meesa", "iam"
    )

    source_entsoe = pd.DataFrame(
        {
            "model": "ENTSO-E",
            "scenario": "TYNDP 2024",
            "region": selected_region,
            "variable": storage_var,
            "unit": "GW",
            "year": sorted(years),
            "value": entsoe_ys,
            "source_type": "benchmark_entsoe",
        }
    )

    source_meesa = pd.DataFrame(
        {
            "model": "MEESA",
            "scenario": FIG4_SCEN_NZ,
            "region": selected_region,
            "variable": storage_var,
            "unit": "GW",
            "year": sorted(years),
            "value": meesa_ys,
            "source_type": "benchmark_meesa_summary",
        }
    )

    source_pypsa = pd.DataFrame(
        {
            "model": "PyPSA-Eur",
            "scenario": "60 weather years",
            "region": selected_region,
            "variable": storage_var,
            "unit": "GW",
            "year": np.arange(1, len(pypsa_values) + 1),
            "value": pypsa_values,
            "source_type": "benchmark_pypsa_distribution",
        }
    )

    source_meta = pd.DataFrame(
        {
            "model": ["Europe installed capacity"],
            "scenario": ["end-2025"],
            "region": [selected_region],
            "variable": [storage_var],
            "unit": ["GW"],
            "year": [2025],
            "value": [EU_INSTALLED_2025_STORAGE_GW],
            "source_type": ["reference_line"],
        }
    )

    source_data = pd.concat(
        [source_iam, source_entsoe, source_meesa, source_pypsa, source_meta],
        ignore_index=True,
        sort=False,
    )

    return fig, source_data