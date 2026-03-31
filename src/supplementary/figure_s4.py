from __future__ import annotations

from src.common.config import (
    SUPPLEMENTARY_OUTPUT_DIR,
    SELECTED_REGION,
    SCENARIOS_DIAGNOSTIC,
    VAR_STORAGE_REL_VRE,
)
from src.common.plot_utils import save_figure
from src.supplementary.supplement_utils import (
    get_default_supplement_subset,
    add_storage_to_vre_ratio,
)


def make_figure_s4(
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
        years=range(2020, 2051, 5),
    )

    df = add_storage_to_vre_ratio(df, storage_var=storage_var)

    ax = df.filter(variable=VAR_STORAGE_REL_VRE).plot.box(
        x="year",
        legend=True,
    )
    ax.set_title(
        "Electricity storage to VRE capacity ratio \nin European net-zero scenarios"
    )
    ax.set_ylabel("Electricity storage relative to \nVRE installed power capacity")

    fig = ax.figure

    if save_png or save_pdf or save_svg:
        save_figure(
            fig,
            SUPPLEMENTARY_OUTPUT_DIR / "Figure-S4",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig
