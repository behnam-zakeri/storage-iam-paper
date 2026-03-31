from __future__ import annotations

from src.common.config import (
    SUPPLEMENTARY_OUTPUT_DIR,
    SELECTED_REGION,
    SCENARIOS_DIAGNOSTIC,
    VAR_STORAGE_POWER,
)
from src.common.plot_utils import save_figure
from src.supplementary.supplement_utils import get_default_supplement_subset


def make_figure_s2(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    selected_region: str = SELECTED_REGION,
    scenarios: list[str] | None = None,
):
    if scenarios is None:
        scenarios = SCENARIOS_DIAGNOSTIC

    df, storage_var = get_default_supplement_subset(
        selected_region=selected_region,
        scenarios=scenarios,
        years=range(2020, 2051),
    )

    ax = df.filter(
        variable=storage_var,
    ).plot(color="model", fill_between=True)

    ax.set_title("Electricity storage capacity in European net-zero scenarios")

    fig = ax.figure

    if save_png or save_pdf or save_svg:
        save_figure(
            fig,
            SUPPLEMENTARY_OUTPUT_DIR / "Figure-S2",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig
