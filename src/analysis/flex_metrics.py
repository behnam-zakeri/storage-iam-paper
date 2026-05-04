from __future__ import annotations

import numpy as np
import pandas as pd
from pyam import IamDataFrame

from src.common.config import (
    SELECTED_SCENARIO,
    SELECTED_REGION,
    VAR_FE_ELEC,
    VAR_FE_TOTAL,
    VAR_ELEC_TOTAL,
    VAR_ELEC_GAS,
    VAR_ELEC_HYDRO,
    VAR_ELEC_BIOMASS,
    VAR_H2_FROM_ELEC,
    VAR_H2_ELEC_CAP,
    VAR_NET_TRADE,
    VAR_ELEC_WIND,
    VAR_ELEC_SOLAR,
    VAR_ELEC_CURT,
    VAR_ELEC_CAPACITY,
    DEMAND_FLEX_SHARE,
    H2_ELECTROLYZER_EFF,
    H2_CAPACITY_FACTOR,
    EJ_PER_GWYR,
    LOAD_FACTOR_EU,
    EJYR_TO_GWAVG,
    DR_TIER_MAP,
    SPATIAL_RESOLUTION_MAP,
    SPATIAL_RES_TO_NREG,
    EUUK_N_REGIONS,
)
from src.analysis.data_loader import load_analysis_data


def _get_year_series(idf: IamDataFrame, variable: str, year: int) -> pd.Series:
    """
    Return a model-indexed Series for one variable and year.
    Duplicate entries are summed within each model.
    """
    sl = idf.filter(variable=variable)
    if sl.empty:
        return pd.Series(dtype=float, name=variable)

    ts = sl.timeseries().reset_index()
    if year not in ts.columns:
        raise ValueError(f"Year {year} not found in data columns for variable '{variable}'.")

    ts[year] = pd.to_numeric(ts[year], errors="coerce")
    s = ts.groupby("model", dropna=False)[year].sum(min_count=1)
    s.name = variable
    return s


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """
    Safe division that replaces zero denominators with NaN.
    """
    return numer / denom.replace(0, np.nan)


def _infer_electrolyzer_capacity_from_h2(
    h2_out_ej_per_yr: pd.Series,
    eff: float = H2_ELECTROLYZER_EFF,
    capacity_factor: float = H2_CAPACITY_FACTOR,
) -> pd.Series:
    """
    Infer electrolyser electrical input capacity (GW) from hydrogen output energy (EJ/yr).

    Logic:
        E_elec_in = E_h2_out / eff
        P_GW = E_elec_in / (CF * EJ_PER_GWYR)
    """
    eff = float(eff)
    capacity_factor = float(capacity_factor)

    if eff <= 0 or eff > 1.0:
        raise ValueError("H2_ELECTROLYZER_EFF must be in (0, 1].")
    if capacity_factor <= 0 or capacity_factor > 1.0:
        raise ValueError("H2_CAPACITY_FACTOR must be in (0, 1].")

    e_elec_in = h2_out_ej_per_yr / eff
    cap_gw = e_elec_in / (capacity_factor * EJ_PER_GWYR)
    cap_gw.name = "Capacity|Hydrogen|Electricity_inferred"
    return cap_gw


def _build_model_annotation_series(idx: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build model-level annotation series for demand response and spatial resolution.
    """
    dr_tier = pd.Series("Unknown", index=idx, name="demand_response")
    for model, tier in DR_TIER_MAP.items():
        if model in dr_tier.index:
            dr_tier.loc[model] = tier

    spatial_res_label = pd.Series("Unknown", index=idx, name="spatial_resolution")
    for model, label in SPATIAL_RESOLUTION_MAP.items():
        if model in spatial_res_label.index:
            spatial_res_label.loc[model] = label

    r_spatial_resolution = (
        spatial_res_label.map(SPATIAL_RES_TO_NREG).astype(float) / EUUK_N_REGIONS
    )
    r_spatial_resolution.name = "r_spatial_resolution"

    return dr_tier, spatial_res_label, r_spatial_resolution


def compute_flex_ratios_all(
    idf: IamDataFrame | pd.DataFrame | None = None,
    scenario: str = SELECTED_SCENARIO,
    region: str = SELECTED_REGION,
    year: int = 2050,
    storage_by_peakload: bool = False,
    demand_flex_share: float = DEMAND_FLEX_SHARE,
) -> pd.DataFrame:
    """
    Compute simplified flexibility indicators for Figure 4a.

    Parameters
    ----------
    idf : IamDataFrame | DataFrame | None
        Input dataset. If None, uses the harmonised analysis dataset from load_analysis_data().
    scenario : str
        Scenario name to analyse.
    region : str
        Region name to analyse.
    year : int
        Target year.
    storage_by_peakload : bool
        If True, storage intensity is computed as storage power / peak load.
        If False, storage intensity is computed as storage power / total electricity capacity.
    demand_flex_share : float
        Stylised flexible share of electrified demand.

    Returns
    -------
    DataFrame
        Model-indexed table of flexibility indicators and diagnostics.
    """
    if idf is None:
        _, eu_nzero, storage_var = load_analysis_data()
        idf = eu_nzero.filter(scenario=scenario, region=region, year=year)
    else:
        if isinstance(idf, pd.DataFrame):
            idf = IamDataFrame(idf)
        _, _, storage_var = load_analysis_data()
        idf = idf.filter(scenario=scenario, region=region, year=year)

    idx = pd.Index(sorted(idf.model), name="model")

    dr_tier, spatial_res_label, r_spatial_resolution = _build_model_annotation_series(idx)

    # Core variables
    storage_power = _get_year_series(idf, storage_var, year).reindex(idx)
    elec_capacity = _get_year_series(idf, VAR_ELEC_CAPACITY, year).reindex(idx)

    elec_total = _get_year_series(idf, VAR_ELEC_TOTAL, year).reindex(idx)
    elec_gas = _get_year_series(idf, VAR_ELEC_GAS, year).reindex(idx)
    elec_hydro = _get_year_series(idf, VAR_ELEC_HYDRO, year).reindex(idx)
    elec_biomass = _get_year_series(idf, VAR_ELEC_BIOMASS, year).reindex(idx)

    fe_elec = _get_year_series(idf, VAR_FE_ELEC, year).reindex(idx)
    fe_total = _get_year_series(idf, VAR_FE_TOTAL, year).reindex(idx)

    # Hydrogen
    h2_from_elec = _get_year_series(idf, VAR_H2_FROM_ELEC, year).reindex(idx)
    h2_cap_reported = _get_year_series(idf, VAR_H2_ELEC_CAP, year).reindex(idx)
    h2_cap_inferred = _infer_electrolyzer_capacity_from_h2(
        h2_out_ej_per_yr=h2_from_elec,
        eff=H2_ELECTROLYZER_EFF,
        capacity_factor=H2_CAPACITY_FACTOR,
    ).reindex(idx)
    h2_cap_used = h2_cap_reported.where(h2_cap_reported.notna(), h2_cap_inferred)
    h2_capacity_was_inferred = h2_cap_reported.isna() & h2_cap_used.notna()

    # Spatial / curtailment
    net_trade = _get_year_series(idf, VAR_NET_TRADE, year).reindex(idx)
    elec_wind = _get_year_series(idf, VAR_ELEC_WIND, year).reindex(idx)
    elec_solar = _get_year_series(idf, VAR_ELEC_SOLAR, year).reindex(idx)
    elec_curt = _get_year_series(idf, VAR_ELEC_CURT, year).reindex(idx)

    # Storage ratio
    if storage_by_peakload:
        peak_load_gw = (fe_elec * EJYR_TO_GWAVG / LOAD_FACTOR_EU).rename(
            "Final Energy|Electricity|Peak"
        )
        r_storage = _safe_div(storage_power, peak_load_gw).rename(
            "r_storage_power_over_peak_load"
        )
        storage_variable_name = "r_storage_power_over_peak_load"
    else:
        r_storage = _safe_div(storage_power, elec_capacity).rename(
            "r_storage_power_over_total_capacity"
        )
        storage_variable_name = "r_storage_power_over_total_capacity"

    # Flexible generation proxy
    flex_gen_numer = elec_gas + elec_hydro + 0.2 * elec_biomass
    r_flexgen = _safe_div(flex_gen_numer, elec_total).rename(
        "r_flexible_gen_over_total_electricity"
    )

    # Flexible demand proxy
    r_electr = _safe_div(fe_elec, fe_total)
    r_flex_demand = (r_electr * demand_flex_share).rename("r_flexible_demand_proxy")

    # Hydrogen flexibility proxy
    r_h2 = _safe_div(h2_cap_used, elec_capacity).rename(
        "r_h2_electrolyzer_capacity_over_total_elec_capacity"
    )

    # Spatial flexibility
    r_trade_net = _safe_div(net_trade.abs(), elec_total).rename("r_net_trade_intensity")

    # Keep this as in your original simplified method
    r_spatial = r_trade_net.rename("r_spatial")

    # Curtailment ratio
    elec_vre = elec_wind + elec_solar
    r_curt = _safe_div(elec_curt, elec_curt + elec_vre).clip(0, 1).rename("r_curt")

    out = pd.DataFrame(
        {
            "scenario": scenario,
            "region": region,
            "year": year,
            "demand_response": dr_tier,
            "spatial_resolution": spatial_res_label,
            storage_variable_name: r_storage,
            "r_flexible_gen_over_total_electricity": r_flexgen,
            "r_flexible_demand_proxy": r_flex_demand,
            "r_h2_electrolyzer_capacity_over_total_elec_capacity": r_h2,
            "r_net_trade_intensity": r_trade_net,
            "r_spatial_resolution": r_spatial_resolution,
            "r_spatial": r_spatial,
            "r_curt": r_curt,
            "h2_electrolyzer_capacity_gw_reported": h2_cap_reported,
            "h2_electrolyzer_capacity_gw_inferred": h2_cap_inferred,
            "h2_electrolyzer_capacity_gw_used": h2_cap_used,
            "h2_capacity_was_inferred": h2_capacity_was_inferred,
        },
        index=idx,
    )

    return out


def get_storage_metric_column(indices: pd.DataFrame) -> str:
    """
    Return the storage metric column name from the flexibility table.
    """
    matches = [c for c in indices.columns if c.startswith("r_storage_power_over_")]
    if not matches:
        raise ValueError("No storage metric column found in flexibility indices table.")
    if len(matches) > 1:
        # Prefer peak-load version only if it is the only non-empty one
        non_empty = [c for c in matches if indices[c].notna().any()]
        if len(non_empty) == 1:
            return non_empty[0]
    return matches[0]


def compute_flexibility_portfolio_diagnostics(
    indices: pd.DataFrame,
    indicator_cols: list[str],
    model_col: str = "model",
    tau: float = 0.03,
    invert_curtailment: bool = False,
) -> pd.DataFrame:
    """
    Compute flexibility portfolio diagnostics for Figure 4b.

    Metrics
    -------
    breadth
        Share of flexibility indicators above threshold tau.

    balance
        Normalized Shannon entropy across flexibility indicators.

    intensity
        Mean normalized indicator value.

    Notes
    -----
    Flexible demand is based on r_flexible_demand_proxy but counted only for
    models with High demand-response representation.
    """
    out = indices.copy()

    if model_col not in out.columns:
        if out.index.name == model_col:
            out = out.reset_index()
        else:
            raise ValueError(f"'{model_col}' not found as column or index.")

    required = [
        "demand_response",
        "r_flexible_demand_proxy",
        *indicator_cols,
    ]
    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(
            "Missing required columns for flexibility portfolio diagnostics: "
            + ", ".join(missing)
        )

    # Flexible demand proxy only counts when demand response is represented as High
    out["r_flexible_demand"] = out["r_flexible_demand_proxy"].where(
        out["demand_response"].astype(str).str.strip().str.lower().eq("high"),
        0.0,
    )

    # Rename to clean diagnostic names
    rename_map = {
        get_storage_metric_column(out): "storage",
        "r_flexible_gen_over_total_electricity": "flexible_generation",
        "r_flexible_demand": "flexible_demand",
        "r_h2_electrolyzer_capacity_over_total_elec_capacity": "hydrogen",
        "r_spatial": "spatial",
        "r_curt": "curtailment",
    }

    out = out.rename(columns=rename_map)

    diagnostic_cols = [
        "storage",
        "flexible_generation",
        "flexible_demand",
        "hydrogen",
        "spatial",
        "curtailment",
    ]

    for col in diagnostic_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(0, 1)

    if invert_curtailment:
        out["curtailment"] = 1 - out["curtailment"]

    x = out[diagnostic_cols].fillna(0.0).clip(0, 1)
    k = len(diagnostic_cols)

    out["breadth"] = (x > tau).sum(axis=1) / k
    out["intensity"] = x.mean(axis=1)

    row_sum = x.sum(axis=1)
    p = x.div(row_sum.replace(0, np.nan), axis=0)

    entropy = -(p * np.log(p)).where(p > 0, 0.0).sum(axis=1)
    out["balance"] = (entropy / np.log(k)).fillna(0.0)

    return out
