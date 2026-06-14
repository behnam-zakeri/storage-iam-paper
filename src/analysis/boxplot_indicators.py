from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import (
    ANALYSIS_OUTPUT_DIR,
    PYPSA_XLSX,
    PYPSA_SHEET,
    ENTSOE_POLICY_DATA,
    FIG4_YEARS,
    FIG4_SCEN_NZ,
    SCENARIOS_DIAGNOSTIC,
    SELECTED_REGION,
    VAR_STORAGE_POWER,
)
from src.analysis.data_loader import load_analysis_data, get_pypsa_storage_row_label


def read_pypsa_storage_distribution(
    pypsa_xlsx: str | Path = PYPSA_XLSX,
    sheet_name: str = PYPSA_SHEET,
) -> np.ndarray:
    """
    Read the PyPSA storage row from the benchmark workbook and return the
    full distribution across weather years as a 1D numpy array.
    """
    pypsa_df = pd.read_excel(pypsa_xlsx, sheet_name=sheet_name)
    pypsa_df["tech"] = pypsa_df["tech"].astype(str)

    storage_row_label = get_pypsa_storage_row_label()
    row = pypsa_df.loc[pypsa_df["tech"].str.strip() == storage_row_label]
    if row.empty:
        raise ValueError(f"Row '{storage_row_label}' not found in sheet '{sheet_name}'.")

    row = row.iloc[0]
    year_cols = [c for c in pypsa_df.columns if c != "tech"]
    return pd.to_numeric(row[year_cols], errors="coerce").dropna().to_numpy()


def prepare_long_table(
    eu_nzero,
    selected_region: str = SELECTED_REGION,
    years: list[int] | tuple[int, ...] = FIG4_YEARS,
    scenarios: list[str] | tuple[str, ...] = SCENARIOS_DIAGNOSTIC,
    storage_var: str = VAR_STORAGE_POWER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare long-format table and an IAM-only subset excluding MEESA.
    Mirrors Figure 3a preparation.
    """
    df = eu_nzero.filter(
        region=selected_region,
        scenario=list(scenarios),
        year=list(years),
        variable=storage_var,
    ).timeseries().reset_index()

    df = df.melt(
        id_vars=["model", "scenario", "region", "variable", "unit"],
        var_name="year",
        value_name="value",
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["year", "value"], inplace=True)
    df["year"] = df["year"].astype(int)

    df_iam = df.loc[df["model"] != "MEESA"].copy()
    return df, df_iam


def _q(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else np.nan


def _pct_rank_cdf(x: float, values: np.ndarray) -> float:
    """Percentile rank in [0, 100] using <= empirical-CDF definition."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0 or np.isnan(x):
        return np.nan
    return 100.0 * (values <= x).mean()


def build_iam_spread_table(df_iam: pd.DataFrame) -> pd.DataFrame:
    """IAM-only spread indicators by scenario × year."""
    g = df_iam.groupby(["scenario", "year"])["value"]

    out = g.agg(
        n_models="count",
        median="median",
        mean="mean",
        std="std",
        min="min",
        max="max",
    ).reset_index()

    qs = (
        g.quantile([0.05, 0.10, 0.25, 0.75, 0.90, 0.95])
        .unstack(level=-1)
        .reset_index()
    )
    qs.columns = ["scenario", "year", "p05", "p10", "p25", "p75", "p90", "p95"]

    out = out.merge(qs, on=["scenario", "year"], how="left")

    out["iqr"] = out["p75"] - out["p25"]
    out["range"] = out["max"] - out["min"]
    out["p95_p05"] = out["p95"] - out["p05"]
    out["p90_p10"] = out["p90"] - out["p10"]
    out["p90_p10_ratio"] = out["p90"] / out["p10"]
    out["p95_p10_ratio"] = out["p95"] / out["p10"]
    out["iqr_over_median"] = out["iqr"] / out["median"]
    out["p95_p05_over_median"] = out["p95_p05"] / out["median"]
    out["p90_p10_over_median"] = out["p90_p10"] / out["median"]

    eps = 1e-6
    tmp = df_iam.copy()
    tmp["is_zero"] = tmp["value"].abs() <= eps
    tmp["lt_10"] = tmp["value"] < 10
    low = tmp.groupby(["scenario", "year"]).agg(
        share_zero=("is_zero", "mean"),
        share_lt_10=("lt_10", "mean"),
    ).reset_index()
    out = out.merge(low, on=["scenario", "year"], how="left")

    num_cols = [
        "median", "mean", "std", "min", "max",
        "p05", "p10", "p25", "p75", "p90", "p95",
        "iqr", "range", "p95_p05", "p90_p10",
        "p90_p10_ratio", "p95_p10_ratio",
        "iqr_over_median", "p95_p05_over_median", "p90_p10_over_median",
        "share_zero", "share_lt_10",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = out[c].round(2)

    return out.sort_values(["year", "scenario"]).reset_index(drop=True)


def build_benchmark_compare_table(
    df_all: pd.DataFrame,
    df_iam: pd.DataFrame,
    years: list[int] | tuple[int, ...],
    meesa_scenario: str,
    entsoe_policy: dict[int, float],
    pypsa_values: np.ndarray,
    benchmark_rank_scenario: str = "NetZero",
) -> pd.DataFrame:
    """
    Compare each benchmark/policy source one-by-one against IAM distribution.
    Benchmarks are not included in the IAM sample.
    """
    rows: list[dict] = []

    def iam_vals(year: int, scen: str = benchmark_rank_scenario) -> np.ndarray:
        return df_iam.loc[
            (df_iam["year"] == year) & (df_iam["scenario"] == scen),
            "value",
        ].to_numpy()

    def iam_summary(year: int, scen: str = benchmark_rank_scenario) -> dict:
        s = df_iam.loc[
            (df_iam["year"] == year) & (df_iam["scenario"] == scen),
            "value",
        ]
        return {
            "iam_median": float(s.median()) if len(s) else np.nan,
            "iam_p10": _q(s, 0.10),
            "iam_p90": _q(s, 0.90),
            "iam_p05": _q(s, 0.05),
            "iam_p95": _q(s, 0.95),
        }

    meesa = (
        df_all[(df_all["model"] == "MEESA") & (df_all["scenario"] == meesa_scenario)]
        .groupby("year", as_index=False)["value"]
        .mean()
    )

    for _, r in meesa.iterrows():
        y = int(r["year"])
        v = float(r["value"])
        vals = iam_vals(y)
        summ = iam_summary(y)
        rows.append({
            "source": "MEESA",
            "detail": meesa_scenario,
            "year": y,
            "value": v,
            "compare_to_iam_scenario": benchmark_rank_scenario,
            "n_iam_rank": int(np.sum(~np.isnan(vals))),
            "iam_median": summ["iam_median"],
            "iam_p10": summ["iam_p10"],
            "iam_p90": summ["iam_p90"],
            "iam_p05": summ["iam_p05"],
            "iam_p95": summ["iam_p95"],
            "pct_rank_in_iam": _pct_rank_cdf(v, vals),
            "delta_vs_iam_median": v - summ["iam_median"],
            "within_iam_p05_p95": (v >= summ["iam_p05"]) and (v <= summ["iam_p95"]),
            "within_iam_p10_p90": (v >= summ["iam_p10"]) and (v <= summ["iam_p90"]),
        })

    for y in years:
        if y not in entsoe_policy:
            continue
        v = float(entsoe_policy[y])
        vals = iam_vals(int(y))
        summ = iam_summary(int(y))
        rows.append({
            "source": "ENTSO-E",
            "detail": "TYNDP 2024",
            "year": int(y),
            "value": v,
            "compare_to_iam_scenario": benchmark_rank_scenario,
            "n_iam_rank": int(np.sum(~np.isnan(vals))),
            "iam_median": summ["iam_median"],
            "iam_p10": summ["iam_p10"],
            "iam_p90": summ["iam_p90"],
            "iam_p05": summ["iam_p05"],
            "iam_p95": summ["iam_p95"],
            "pct_rank_in_iam": _pct_rank_cdf(v, vals),
            "delta_vs_iam_median": v - summ["iam_median"],
            "within_iam_p05_p95": (v >= summ["iam_p05"]) and (v <= summ["iam_p95"]),
            "within_iam_p10_p90": (v >= summ["iam_p10"]) and (v <= summ["iam_p90"]),
        })

    py = np.asarray(pypsa_values, dtype=float)
    py = py[~np.isnan(py)]
    if py.size:
        y = 2050
        vals = iam_vals(y)
        summ = iam_summary(y)
        py_med = float(np.median(py))
        py_p10 = float(np.quantile(py, 0.10))
        py_p90 = float(np.quantile(py, 0.90))
        py_p05 = float(np.quantile(py, 0.05))
        py_p95 = float(np.quantile(py, 0.95))

        rows.append({
            "source": "PyPSA-Eur",
            "detail": "net-zero snapshot; weather-year distribution",
            "year": y,
            "value": py_med,
            "value_p10": py_p10,
            "value_p90": py_p90,
            "value_p05": py_p05,
            "value_p95": py_p95,
            "compare_to_iam_scenario": benchmark_rank_scenario,
            "n_iam_rank": int(np.sum(~np.isnan(vals))),
            "iam_median": summ["iam_median"],
            "iam_p10": summ["iam_p10"],
            "iam_p90": summ["iam_p90"],
            "iam_p05": summ["iam_p05"],
            "iam_p95": summ["iam_p95"],
            "pct_rank_in_iam": _pct_rank_cdf(py_med, vals),
            "delta_vs_iam_median": py_med - summ["iam_median"],
            "within_iam_p05_p95": (py_med >= summ["iam_p05"]) and (py_med <= summ["iam_p95"]),
            "within_iam_p10_p90": (py_med >= summ["iam_p10"]) and (py_med <= summ["iam_p90"]),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["year", "source"]).reset_index(drop=True)

    for c in [
        "value", "value_p10", "value_p90", "value_p05", "value_p95",
        "iam_median", "iam_p10", "iam_p90", "iam_p05", "iam_p95",
        "pct_rank_in_iam", "delta_vs_iam_median",
    ]:
        if c in out.columns:
            out[c] = out[c].round(2)

    return out


def build_inset_table_netzero(
    iam_spread_tbl: pd.DataFrame,
    bench_tbl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact inset table for Figure 3a, NetZero only.
    """
    nz = iam_spread_tbl[iam_spread_tbl["scenario"] == "NetZero"].copy()
    nz = nz.set_index("year")

    nz["iam_med_band"] = nz.apply(
        lambda r: f"{r['median']:.0f} [{r['p10']:.0f}–{r['p90']:.0f}]",
        axis=1,
    )
    nz["iqr_str"] = nz["iqr"].map(lambda x: f"{x:.0f}")

    b = bench_tbl.pivot_table(index="year", columns="source", values="value", aggfunc="first")

    py = bench_tbl[bench_tbl["source"] == "PyPSA-Eur"].set_index("year")
    py_str = pd.Series(index=nz.index, dtype=object)
    if 2050 in py.index:
        if "value_p10" in py.columns and "value_p90" in py.columns:
            py_str.loc[2050] = (
                f"{py.loc[2050, 'value']:.0f} "
                f"[{py.loc[2050, 'value_p10']:.0f}–{py.loc[2050, 'value_p90']:.0f}]"
            )
        else:
            py_str.loc[2050] = f"{py.loc[2050, 'value']:.0f}"
    py_str = py_str.fillna("—")

    years = list(nz.index)

    return pd.DataFrame({
        "Year": years,
        "IAM NetZero median [p10–p90] (GW)": [nz.loc[y, "iam_med_band"] for y in years],
        "IQR (GW)": [nz.loc[y, "iqr_str"] for y in years],
        "MEESA (GW)": [
            f"{b.loc[y, 'MEESA']:.0f}" if ("MEESA" in b.columns and y in b.index and pd.notna(b.loc[y, "MEESA"])) else "—"
            for y in years
        ],
        "ENTSO-E (GW)": [
            f"{b.loc[y, 'ENTSO-E']:.0f}" if ("ENTSO-E" in b.columns and y in b.index and pd.notna(b.loc[y, "ENTSO-E"])) else "—"
            for y in years
        ],
        "PyPSA (snapshot) (GW)": [py_str.loc[y] for y in years],
    })


def make_analysis1_tables(
    save_csv: bool = True,
    out_dir: str | Path = ANALYSIS_OUTPUT_DIR,
    pypsa_xlsx: str | Path = PYPSA_XLSX,
    pypsa_sheet: str = PYPSA_SHEET,
    selected_region: str = SELECTED_REGION,
    years: list[int] | tuple[int, ...] = FIG4_YEARS,
    scenarios: list[str] | tuple[str, ...] = SCENARIOS_DIAGNOSTIC,
    storage_var: str | None = None,
    benchmark_rank_scenario: str = FIG4_SCEN_NZ,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build and optionally save Analysis 1 tables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        iam_spread_tbl, benchmark_compare_tbl, inset_tbl
    """
    out_dir = Path(out_dir)
    if save_csv:
        out_dir.mkdir(parents=True, exist_ok=True)

    _, eu_nzero, storage_var_loaded = load_analysis_data()
    storage_var = storage_var or storage_var_loaded or VAR_STORAGE_POWER

    pypsa_values = read_pypsa_storage_distribution(
        pypsa_xlsx=pypsa_xlsx,
        sheet_name=pypsa_sheet,
    )

    df_all, df_iam = prepare_long_table(
        eu_nzero=eu_nzero,
        selected_region=selected_region,
        years=years,
        scenarios=scenarios,
        storage_var=storage_var,
    )

    iam_spread_tbl = build_iam_spread_table(df_iam)
    bench_tbl = build_benchmark_compare_table(
        df_all=df_all,
        df_iam=df_iam,
        years=years,
        meesa_scenario=FIG4_SCEN_NZ,
        entsoe_policy=dict(ENTSOE_POLICY_DATA),
        pypsa_values=pypsa_values,
        benchmark_rank_scenario=benchmark_rank_scenario,
    )
    inset_tbl = build_inset_table_netzero(iam_spread_tbl, bench_tbl)

    if save_csv:
        iam_spread_tbl.to_csv(out_dir / "analysis1_iam_spread.csv", index=False)
        bench_tbl.to_csv(out_dir / "analysis1_benchmarks_vs_iam.csv", index=False)
        inset_tbl.to_csv(out_dir / "analysis1_inset_table_netzero.csv", index=False)

    return iam_spread_tbl, bench_tbl, inset_tbl


# Backward-compatible script-style alias.
def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_analysis1_tables(save_csv=True)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)

    iam_spread_tbl, bench_tbl, inset_tbl = main()

    print("\n=== Analysis 1: IAM spread table (scenario × year) ===")
    print(iam_spread_tbl)

    print("\n=== Analysis 1: Benchmarks/policy vs IAM (one-by-one) ===")
    print(bench_tbl)

    print("\n=== Analysis 1: Compact inset table (NetZero only) ===")
    print(inset_tbl)
