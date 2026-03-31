from __future__ import annotations

import pandas as pd
from pyam import IamDataFrame

from src.common.config import (
    FIG4_SCEN_NZ,
    FIG4_SCEN_DIAG,
    FIG4_SCEN_FALLBACK,
    FIG4_YEARS,
    SELECTED_REGION,
    VAR_ELEC_TOTAL,
    VAR_ELEC_WIND,
    VAR_ELEC_SOLAR,
    VAR_ELEC_VRE,
    VAR_CAP_WIND,
    VAR_CAP_SOLAR,
    VAR_CAP_VRE,
    VAR_ELEC_CAPACITY,
    VAR_FE_ELEC,
    VAR_FE_ELEC_PEAK,
    VAR_VRE_SHARE,
    VAR_STORAGE_REL_TOTAL,
    VAR_STORAGE_REL_VRE,
    VAR_STORAGE_REL_PEAK,
    EJYR_TO_GWAVG,
    LOAD_FACTOR_ANALYSIS,
)
from src.analysis.data_loader import load_analysis_data


def get_default_analysis_subset(
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    scenario_set: list[str] | None = None,
) -> tuple[IamDataFrame, str]:
    """
    Load the harmonised IAM analysis dataset and return the default subset
    used in the Figure 4 scatter workflows.

    Returns
    -------
    df : IamDataFrame
        Filtered subset.
    storage_var : str
        Storage variable to use downstream.
    """
    if years is None:
        years = FIG4_YEARS
    if scenario_set is None:
        scenario_set = [FIG4_SCEN_NZ, FIG4_SCEN_DIAG, FIG4_SCEN_FALLBACK]

    _, eu_nzero, storage_var = load_analysis_data()
    df = eu_nzero.filter(
        region=selected_region,
        scenario=scenario_set,
        year=years,
    )
    return df, storage_var


def add_vre_generation(df: IamDataFrame) -> IamDataFrame:
    """
    Add VRE generation as wind + solar.
    """
    out = df.copy()
    out.add(
        VAR_ELEC_WIND,
        VAR_ELEC_SOLAR,
        VAR_ELEC_VRE,
        append=True,
    )
    return out


def add_vre_share(df: IamDataFrame) -> IamDataFrame:
    """
    Add VRE share in total electricity generation.
    """
    out = add_vre_generation(df)
    out.divide(
        VAR_ELEC_VRE,
        VAR_ELEC_TOTAL,
        VAR_VRE_SHARE,
        append=True,
    )
    return out


def add_vre_capacity(df: IamDataFrame) -> IamDataFrame:
    """
    Add VRE capacity as wind + solar capacity.
    """
    out = df.copy()
    out.add(
        VAR_CAP_WIND,
        VAR_CAP_SOLAR,
        VAR_CAP_VRE,
        append=True,
    )
    return out


def add_storage_to_total_capacity_ratio(
    df: IamDataFrame,
    storage_var: str,
) -> IamDataFrame:
    """
    Add storage power relative to total installed electricity capacity.
    """
    out = df.copy()
    out.divide(
        storage_var,
        VAR_ELEC_CAPACITY,
        VAR_STORAGE_REL_TOTAL,
        append=True,
    )
    return out


def add_storage_to_vre_capacity_ratio(
    df: IamDataFrame,
    storage_var: str,
) -> IamDataFrame:
    """
    Add storage power relative to VRE installed capacity.
    """
    out = add_vre_capacity(df)
    out.divide(
        storage_var,
        VAR_CAP_VRE,
        VAR_STORAGE_REL_VRE,
        append=True,
    )
    return out


def add_peak_load_proxy(
    df: IamDataFrame,
    load_factor: float = LOAD_FACTOR_ANALYSIS,
) -> IamDataFrame:
    """
    Add peak load proxy based on annual electricity demand.

    1 EJ/yr ≈ 31.7 GW average
    GW_peak = GW_avg / load_factor
    """
    out = df.copy()
    df_el = out.filter(variable=VAR_FE_ELEC).timeseries()
    df_el *= EJYR_TO_GWAVG * (1.0 / load_factor)
    df_el = df_el.reset_index()
    df_el["variable"] = VAR_FE_ELEC_PEAK
    out = out.append(df_el)
    return out


def add_storage_to_peak_ratio(
    df: IamDataFrame,
    storage_var: str,
    load_factor: float = LOAD_FACTOR_ANALYSIS,
) -> IamDataFrame:
    """
    Add storage power relative to peak load.
    """
    out = add_peak_load_proxy(df, load_factor=load_factor)
    out.divide(
        storage_var,
        VAR_FE_ELEC_PEAK,
        VAR_STORAGE_REL_PEAK,
        append=True,
    )
    return out


def build_wide_pair_table(
    df: IamDataFrame,
    var_x: str,
    var_y: str,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a wide plotting table with one row per (model, scenario, year)
    and two value columns: var_x and var_y.
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


def apply_diagnostic_fallback(
    df_wide: pd.DataFrame,
    scen_nz: str = FIG4_SCEN_NZ,
    scen_diag: str = FIG4_SCEN_DIAG,
    scen_fallback: str = FIG4_SCEN_FALLBACK,
) -> pd.DataFrame:
    """
    Use fallback scenario only for models missing the diagnostic scenario.
    Then unify plotting style/label under the diagnostic key.
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
        out["scenario"] == scen_nz,
        scen_diag,
    )
    out["scenario_source"] = out["scenario"]
    return out


def build_storage_total_scatter_data(
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    scenario_set: list[str] | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """
    Prepare the underlying data for the storage-to-total-capacity scatter plot.
    """
    df, storage_var = get_default_analysis_subset(
        selected_region=selected_region,
        years=years,
        scenario_set=scenario_set,
    )
    df = add_vre_share(df)
    df = add_storage_to_total_capacity_ratio(df, storage_var=storage_var)

    var_x = VAR_VRE_SHARE
    var_y = VAR_STORAGE_REL_TOTAL

    df_wide = build_wide_pair_table(df, var_x=var_x, var_y=var_y, years=years)
    df_wide = apply_diagnostic_fallback(df_wide)
    return df_wide, var_x, var_y


def build_storage_vre_scatter_data(
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    scenario_set: list[str] | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """
    Prepare the underlying data for the storage-to-VRE-capacity scatter plot.
    """
    df, storage_var = get_default_analysis_subset(
        selected_region=selected_region,
        years=years,
        scenario_set=scenario_set,
    )
    df = add_vre_share(df)
    df = add_storage_to_vre_capacity_ratio(df, storage_var=storage_var)

    var_x = VAR_VRE_SHARE
    var_y = VAR_STORAGE_REL_VRE

    df_wide = build_wide_pair_table(df, var_x=var_x, var_y=var_y, years=years)
    df_wide = apply_diagnostic_fallback(df_wide)
    return df_wide, var_x, var_y


def build_storage_peak_scatter_data(
    selected_region: str = SELECTED_REGION,
    years: list[int] | None = None,
    scenario_set: list[str] | None = None,
    load_factor: float = LOAD_FACTOR_ANALYSIS,
) -> tuple[pd.DataFrame, str, str]:
    """
    Prepare the underlying data for Figure 4b:
    storage relative to peak load vs VRE share.
    """
    df, storage_var = get_default_analysis_subset(
        selected_region=selected_region,
        years=years,
        scenario_set=scenario_set,
    )
    df = add_vre_share(df)
    df = add_storage_to_peak_ratio(
        df,
        storage_var=storage_var,
        load_factor=load_factor,
    )

    var_x = VAR_VRE_SHARE
    var_y = VAR_STORAGE_REL_PEAK

    df_wide = build_wide_pair_table(df, var_x=var_x, var_y=var_y, years=years)
    df_wide = apply_diagnostic_fallback(df_wide)
    return df_wide, var_x, var_y