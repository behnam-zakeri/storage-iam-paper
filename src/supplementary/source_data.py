from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import SUPPLEMENTARY_DATA_DIR
from src.common.io_utils import write_excel_sheets


SUPPLEMENTARY_SOURCE_DATA_XLSX = SUPPLEMENTARY_DATA_DIR / "Figures-S_source-data.xlsx"


def export_supplementary_source_data(
    data_tables: dict[str, pd.DataFrame],
    output_file: str | Path = SUPPLEMENTARY_SOURCE_DATA_XLSX,
):
    """
    Export supplementary figure source data to one Excel workbook.

    Parameters
    ----------
    data_tables : dict[str, DataFrame]
        Mapping from sheet name to source-data table.
    output_file : str | Path
        Output workbook path.
    """
    clean_tables = {
        sheet_name: df.copy()
        for sheet_name, df in data_tables.items()
        if df is not None and not df.empty
    }

    write_excel_sheets(output_file, sheets=clean_tables, index=False)