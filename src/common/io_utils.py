from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path) -> Path:
    """
    Create directory if it does not exist and return it as Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(x):
    """
    Collapse whitespace and normalise line breaks.
    Keeps NaN values untouched.
    """
    if pd.isna(x):
        return x
    s = str(x).replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_excel_sheet(xlsx_file, sheet_name: str) -> pd.DataFrame:
    """
    Read one Excel sheet and standardise column names.
    """
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    return standardise_columns(df)


def read_csv(csv_file, **kwargs) -> pd.DataFrame:
    """
    Read CSV and standardise column names.
    """
    df = pd.read_csv(csv_file, **kwargs)
    return standardise_columns(df)


def ensure_required_columns(df: pd.DataFrame, required: list[str], context: str = ""):
    """
    Raise a clear error if required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        ctx = f" in {context}" if context else ""
        raise ValueError(f"Missing required columns{ctx}: {missing}")


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert selected columns to numeric, coercing invalid values to NaN.
    """
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def first_existing_column(df: pd.DataFrame, candidates: list[str], context: str = "") -> str:
    """
    Return the first matching column from a list of possible names.
    """
    for c in candidates:
        if c in df.columns:
            return c
    ctx = f" in {context}" if context else ""
    raise ValueError(f"None of the candidate columns found{ctx}: {candidates}")


def write_excel_sheets(output_file, sheets: dict[str, pd.DataFrame], index: bool = False):
    """
    Write multiple DataFrames to one Excel workbook.

    Parameters
    ----------
    output_file : str | Path
        Output workbook path.
    sheets : dict[str, DataFrame]
        Sheet name -> DataFrame mapping.
    index : bool
        Whether to write DataFrame index.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=index)


def melt_year_columns(
    df: pd.DataFrame,
    year_cols: list,
    id_vars: list[str],
    var_name: str = "year",
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Melt wide year columns into long format and coerce year to int where possible.
    """
    out = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name=var_name,
        value_name=value_name,
    ).copy()

    out[var_name] = pd.to_numeric(out[var_name], errors="coerce")
    return out


def reorder_columns(df: pd.DataFrame, first_cols: list[str]) -> pd.DataFrame:
    """
    Move selected columns to the front, preserving the order of the rest.
    """
    existing_first = [c for c in first_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_first]
    return df[existing_first + remaining].copy()