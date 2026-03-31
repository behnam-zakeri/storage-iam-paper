from __future__ import annotations

from src.common.config import SELECTED_REGION
from src.common.plot_utils import save_figure
from src.supplementary.supplement_utils import (
    get_default_supplement_subset,
    add_vre_generation_and_share,
    add_storage_to_total_ratio,
    filter_out_scenarios,
    build_yearly_scatter_table,
    plot_supplement_scatter_by_year,
)


def make_figure_s6(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    output_dir=None,
    selected_region: str = SELECTED_REGION,
):
    """
    Figure S6:
    Relative capacity of storage vs share of VRE in net-zero scenarios.
    """
    years = [2030, 2040, 2050]

    df, storage_var = get_default_supplement_subset(
        selected_region=selected_region,
        years=years,
    )

    df = add_vre_generation_and_share(df)
    df = add_storage_to_total_ratio(df, storage_var=storage_var)
    df = filter_out_scenarios(df, exclude_patterns=["*High-eff", "*e-fuel"])

    var_x = "Share in Secondary Energy|Electricity|VRE"
    var_y = "Relative Capacity|Electricity|Storage to Total"

    df_plot = build_yearly_scatter_table(
        df,
        var_x=var_x,
        var_y=var_y,
        years=years,
    )

    fig, ax = plot_supplement_scatter_by_year(
        df_plot,
        var_x=var_x,
        var_y=var_y,
        xlim=(0, 1),
        ylim=(0, 0.4),
        xlabel="Share of VRE from total electricity generation",
        ylabel="Storage capacity relative to \ntotal installed power capacity",
        title="Relative capacity of storage vs. share of VRE in \nnet-zero scenarios for Europe (2030, 2040, and 2050)",
        marker_size=100,
        legend_frameon=False,
    )

    if output_dir is not None and (save_png or save_pdf or save_svg):
        save_figure(
            fig,
            output_dir / "Figure-S6",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, df_plot

