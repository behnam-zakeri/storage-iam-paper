from __future__ import annotations

from src.common.config import SELECTED_REGION
from src.common.plot_utils import save_figure
from src.supplementary.supplement_utils import (
    get_default_supplement_subset,
    add_flexible_capacity_share,
    add_storage_to_total_ratio,
    filter_out_scenarios,
    build_yearly_scatter_table,
    plot_supplement_scatter_by_year,
)


def make_figure_s7(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    output_dir=None,
    selected_region: str = SELECTED_REGION,
):
    """
    Figure S7:
    Relative capacity of storage vs flexible generation capacity share.
    """
    years = [2030, 2040, 2050]

    df, storage_var = get_default_supplement_subset(
        selected_region=selected_region,
        years=years,
    )

    df = add_flexible_capacity_share(df)
    df = add_storage_to_total_ratio(df, storage_var=storage_var)
    df = filter_out_scenarios(df, exclude_patterns=["*High-eff", "*e-fuel"])

    var_x = "Relative Capacity|Electricity|Flexible to Total"
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
        ylim=(0, 0.5),
        xlabel="Flexible generation capacity relative to \ntotal installed power capacity",
        ylabel="Storage capacity relative to \ntotal installed power capacity",
        title="Relative capacity of storage vs. flexible generation in \nnet-zero scenarios for Europe (2030, 2040, and 2050)",
        marker_size=100,
        legend_frameon=False,
    )

    if output_dir is not None and (save_png or save_pdf or save_svg):
        save_figure(
            fig,
            output_dir / "Figure-S7",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, df_plot