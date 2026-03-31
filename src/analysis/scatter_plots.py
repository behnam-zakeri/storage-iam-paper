from __future__ import annotations

import numpy as np
import pandas as pd
from pyam import IamDataFrame
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from src.common.config import (
    ANALYSIS_OUTPUT_DIR,
    PYPSA_XLSX,
    PYPSA_SHEET,
    SCENARIO_COLORS,
    FIG4_SCEN_NZ,
    FIG4_SCEN_DIAG,
    FIG4_SCEN_FALLBACK,
    FIG4_YEARS,
    SELECTED_REGION,
    LOAD_FACTOR_ANALYSIS,
    EJYR_TO_GWAVG,
    VAR_STORAGE_POWER,
    VAR_ELEC_TOTAL,
    VAR_ELEC_WIND,
    VAR_ELEC_SOLAR,
    VAR_ELEC_VRE,
    VAR_FE_ELEC,
    VAR_FE_ELEC_PEAK,
    VAR_VRE_SHARE,
    VAR_STORAGE_REL_PEAK,
)
from src.common.plot_utils import (
    save_figure,
    horizontal_grid_only,
    scenario_path_legend_handles,
)
from src.analysis.data_loader import load_analysis_data, build_model_markers


# =========================================================
# Shared data preparation
# =========================================================
def _prepare_base_df(
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    scenario_set: list[str] | None = None,
) -> IamDataFrame:
    """
    Load the harmonised IAMC dataset and filter to the requested region/scenarios/years.
    """
    if years is None:
        years = FIG4_YEARS
    if scenario_set is None:
        scenario_set = [FIG4_SCEN_NZ, FIG4_SCEN_DIAG, FIG4_SCEN_FALLBACK]

    eu_nzero, _, _ = load_analysis_data()

    df = eu_nzero.filter(
        region=selected_region,
        scenario=scenario_set,
        year=years,
    )

    return df


def _add_vre_generation_and_share(df: IamDataFrame) -> IamDataFrame:
    """
    Add VRE generation and its share in total electricity generation.
    """
    df = df.copy()

    df.add(
        VAR_ELEC_WIND,
        VAR_ELEC_SOLAR,
        VAR_ELEC_VRE,
        append=True,
    )

    df.divide(
        VAR_ELEC_VRE,
        VAR_ELEC_TOTAL,
        VAR_VRE_SHARE,
        append=True,
    )
    return df


def _add_peak_load_and_storage_ratio(
    df: IamDataFrame,
    load_factor: float = LOAD_FACTOR_ANALYSIS,
) -> IamDataFrame:
    """
    Infer peak electricity demand from annual Final Energy|Electricity and compute
    storage power capacity relative to peak load.

    1 EJ/yr ≈ 31.7 GW average
    GW_peak = GW_avg / load_factor
    """
    df = df.copy()

    df_el = df.filter(variable=VAR_FE_ELEC).timeseries()
    df_el *= EJYR_TO_GWAVG * (1.0 / load_factor)
    df_el = df_el.reset_index()
    df_el["variable"] = VAR_FE_ELEC_PEAK
    df = df.append(df_el)

    df.divide(
        VAR_STORAGE_POWER,
        VAR_FE_ELEC_PEAK,
        VAR_STORAGE_REL_PEAK,
        append=True,
    )
    return df


def _wide_pair_table(
    df: IamDataFrame,
    var_x: str,
    var_y: str,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build wide plotting table with one row per (model, scenario, year).
    """
    if years is None:
        years = FIG4_YEARS

    df_sub = df.filter(variable=[var_x, var_y])

    ts = df_sub.timeseries().reset_index()
    ts_long = ts.melt(
        id_vars=[c for c in ts.columns if c not in years],
        value_vars=years,
        var_name="year",
        value_name="value",
    )
    ts_long["year"] = ts_long["year"].astype(int)

    df_wide = ts_long.pivot_table(
        index=["model", "scenario", "year"],
        columns="variable",
        values="value",
        aggfunc="mean",
    ).reset_index()

    return df_wide


def _apply_diag_fallback(
    df_wide: pd.DataFrame,
    scen_nz: str = FIG4_SCEN_NZ,
    scen_diag: str = FIG4_SCEN_DIAG,
    scen_fallback: str = FIG4_SCEN_FALLBACK,
) -> pd.DataFrame:
    """
    Use fallback scenario only for models missing the diagnostic scenario,
    then unify plotting style under the diagnostic key.
    """
    models_with_diag = set(
        df_wide.loc[df_wide["scenario"] == scen_diag, "model"].unique()
    )

    keep = (
        (df_wide["scenario"] == scen_nz)
        | (df_wide["scenario"] == scen_diag)
        | (
            (df_wide["scenario"] == scen_fallback)
            & (~df_wide["model"].isin(models_with_diag))
        )
    )

    out = df_wide.loc[keep].copy()

    out["scenario_plot"] = out["scenario"].where(
        out["scenario"] == scen_nz, scen_diag
    )
    out["scenario_source"] = out["scenario"]

    return out


def _scenario_path_plot(
    df_wide: pd.DataFrame,
    var_x: str,
    var_y: str,
    title: str | None,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    outfile_stem: str | None = None,
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    extra_legend_handles: list | None = None,
):
    """
    Shared scatter-path plot:
    - line across years
    - same model marker across years
    - 2050 bold black edge
    """
    _, eu_nzero, _ = load_analysis_data()
    model_markers = build_model_markers(eu_nzero)

    fig, ax = plt.subplots(figsize=(7, 5))

    for (model, scen_key), g in df_wide.groupby(["model", "scenario_plot"]):
        g = g.sort_values("year")

        ax.plot(
            g[var_x],
            g[var_y],
            color=SCENARIO_COLORS[scen_key],
            linewidth=0.5,
            alpha=0.55,
            zorder=1,
        )

        g_normal = g[g["year"] != 2050]
        g_2050 = g[g["year"] == 2050]

        ax.scatter(
            g_normal[var_x],
            g_normal[var_y],
            marker=model_markers.get(model, "o"),
            color=SCENARIO_COLORS[scen_key],
            s=50,
            edgecolor="grey",
            linewidths=0.8,
            zorder=2,
        )

        ax.scatter(
            g_2050[var_x],
            g_2050[var_y],
            marker=model_markers.get(model, "o"),
            color=SCENARIO_COLORS[scen_key],
            s=50,
            edgecolor="black",
            linewidths=1.2,
            zorder=3,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, pad=20)

    horizontal_grid_only(ax, linewidth=0.8, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)

    handles = scenario_path_legend_handles(
        models=sorted(df_wide["model"].unique()),
        model_markers=model_markers,
        scenario_colors=SCENARIO_COLORS,
        scen_nz=FIG4_SCEN_NZ,
        scen_diag=FIG4_SCEN_DIAG,
        include_benchmarks=extra_legend_handles,
    )
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    if outfile_stem:
        save_figure(
            fig,
            ANALYSIS_OUTPUT_DIR / outfile_stem,
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, ax


# =========================================================
# PyPSA helper
# =========================================================
def _read_pypsa_ellipse_data() -> dict[str, object]:
    """
    Read PyPSA benchmark arrays and precompute ellipse parameters.
    """
    pypsa_df = pd.read_excel(PYPSA_XLSX, sheet_name=PYPSA_SHEET)

    var_x_label = "Share of VRE from total generation"
    var_y_label = "Electricity storage to peak demand"

    year_cols = [c for c in pypsa_df.columns if c != "tech"]

    x_row = pypsa_df.loc[pypsa_df["tech"].astype(str).str.strip() == var_x_label]
    y_row = pypsa_df.loc[pypsa_df["tech"].astype(str).str.strip() == var_y_label]

    if x_row.empty:
        raise ValueError(f"Row '{var_x_label}' not found in sheet '{PYPSA_SHEET}'.")
    if y_row.empty:
        raise ValueError(f"Row '{var_y_label}' not found in sheet '{PYPSA_SHEET}'.")

    x = pd.to_numeric(x_row.iloc[0][year_cols], errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(y_row.iloc[0][year_cols], errors="coerce").dropna().to_numpy()

    mu = np.array([x.mean(), y.mean()])
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    scale = 2.447  # ~95% ellipse for 2D normal
    width, height = 2 * scale * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    return {
        "x": x,
        "y": y,
        "mu": mu,
        "width": width,
        "height": height,
        "angle": angle,
        "x_median": float(np.median(x)),
        "y_median": float(np.median(y)),
    }


# =========================================================
# Figure 4b
# =========================================================
def make_figure4b(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    add_pypsa: bool = True,
):
    """
    Manuscript Figure 4b:
    Electricity storage relative to peak demand vs VRE share.
    """
    _, eu_nzero, _ = load_analysis_data()
    model_markers = build_model_markers(eu_nzero)

    df = _prepare_base_df()
    df = _add_vre_generation_and_share(df)
    df = _add_peak_load_and_storage_ratio(df)

    var_x = VAR_VRE_SHARE
    var_y = VAR_STORAGE_REL_PEAK

    df_wide = _wide_pair_table(df, var_x=var_x, var_y=var_y)
    df_wide = _apply_diag_fallback(df_wide)

    fig, ax = plt.subplots(figsize=(7, 5))

    for (model, scen_key), g in df_wide.groupby(["model", "scenario_plot"]):
        g = g.sort_values("year")

        ax.plot(
            g[var_x],
            g[var_y],
            color=SCENARIO_COLORS[scen_key],
            linewidth=0.5,
            alpha=0.55,
            zorder=1,
        )

        g_normal = g[g["year"] != 2050]
        g_2050 = g[g["year"] == 2050]

        ax.scatter(
            g_normal[var_x],
            g_normal[var_y],
            marker=model_markers.get(model, "o"),
            color=SCENARIO_COLORS[scen_key],
            s=50,
            edgecolor="grey",
            linewidths=0.8,
            zorder=2,
        )

        ax.scatter(
            g_2050[var_x],
            g_2050[var_y],
            marker=model_markers.get(model, "o"),
            color=SCENARIO_COLORS[scen_key],
            s=50,
            edgecolor="black",
            linewidths=1.2,
            zorder=3,
        )

    # cluster ellipses
    cluster_ellipses = [
        dict(
            xy=(0.63, 0.10),
            width=0.70,
            height=0.27,
            angle=15,
            alpha=0.12,
            label="Lower storage share",
            text_xy=(0.85, 0.30),
        ),
        dict(
            xy=(0.60, 0.42),
            width=1.25,
            height=0.24,
            angle=60,
            alpha=0.12,
            label="Higher storage share",
            text_xy=(0.70, 0.95),
        ),
    ]

    for e in cluster_ellipses:
        ax.add_patch(
            Ellipse(
                xy=e["xy"],
                width=e["width"],
                height=e["height"],
                angle=e["angle"],
                facecolor="grey",
                edgecolor="grey",
                linewidth=1.0,
                alpha=e["alpha"],
                zorder=1,
            )
        )
        ax.text(
            e["text_xy"][0],
            e["text_xy"][1],
            e["label"],
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            zorder=4,
        )

    pypsa_payload = None
    if add_pypsa:
        pypsa_payload = _read_pypsa_ellipse_data()

        ell = Ellipse(
            pypsa_payload["mu"],
            pypsa_payload["width"],
            pypsa_payload["height"],
            angle=pypsa_payload["angle"],
            alpha=0.35,
            zorder=3,
        )
        ax.add_patch(ell)

        ax.scatter(
            pypsa_payload["x_median"],
            pypsa_payload["y_median"],
            marker="*",
            s=12,
            zorder=4,
            edgecolor="black",
        )

        ax.text(
            pypsa_payload["x_median"] - 0.04,
            pypsa_payload["y_median"] + 0.03,
            "PyPSA-Eur",
            ha="left",
            va="center",
            fontsize=9,
            color="grey",
            zorder=8,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        1.08,
        "b) Electricity storage deployment vs. VRE integration",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.5,
        1.03,
        "(EU27 & UK, 2030–2050)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
    )

    ax.set_xlabel("Share of VRE from total electricity generation (TWh/TWh)")
    ax.set_ylabel("Electricity storage capacity \nrelative to peak demand (GW/GW)")

    horizontal_grid_only(ax, linewidth=0.8, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)

    benchmark_handles = [
        Line2D(
            [0], [0],
            marker=model_markers.get("MEESA", "^"),
            color="w",
            label="MEESA",
            markerfacecolor="gray",
            markeredgecolor="grey",
            markersize=7,
        ),
        Line2D(
            [0], [0],
            marker="*",
            color="w",
            label="PyPSA-Eur (60 runs)",
            markerfacecolor="gray",
            markeredgecolor="grey",
            markersize=7,
        ),
    ]

    model_list = [
        x for x in sorted(df_wide["model"].unique())
        if x not in ["MEESA", "TIAM-ECN"]
    ]

    handles = scenario_path_legend_handles(
        models=model_list,
        model_markers=model_markers,
        scenario_colors=SCENARIO_COLORS,
        scen_nz=FIG4_SCEN_NZ,
        scen_diag=FIG4_SCEN_DIAG,
        include_benchmarks=benchmark_handles,
    )
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    if save_png or save_pdf or save_svg:
        save_figure(
            fig,
            ANALYSIS_OUTPUT_DIR / "Figure-4b",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    # source data
    data_out = df_wide.copy()
    if pypsa_payload is not None:
        pypsa_points = pd.DataFrame(
            {
                "model": "PyPSA-Eur",
                "scenario": "historical weather years",
                "year": np.arange(len(pypsa_payload["x"])) + 1,
                var_x: pypsa_payload["x"],
                var_y: pypsa_payload["y"],
            }
        )
        data_out = pd.concat([data_out, pypsa_points], ignore_index=True)

    return fig, data_out