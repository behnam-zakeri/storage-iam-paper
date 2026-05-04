from __future__ import annotations

from src.common.config import FIG4_SOURCE_DATA_XLSX
from src.common.io_utils import write_excel_sheets


def export_figure4_source_data(
    figure3a,
    figure3b,
    figure4a,
    figure4b,
    output_file=FIG4_SOURCE_DATA_XLSX,
):
    """
    Export source data for manuscript Figure 4 panels to one Excel workbook.

    Parameters
    ----------
    figure3a, figure3b, figure4a, figure 4b : pandas.DataFrame
        Source-data tables for each panel.
    output_file : str or Path
        Target Excel workbook.
    """
    sheets = {
        "Figure-3a": figure3a,
        "Figure-3b": figure3b,
        "Figure-4a": figure4a,
        "Figure-4b": figure4b,
    }
    write_excel_sheets(output_file, sheets=sheets, index=False)