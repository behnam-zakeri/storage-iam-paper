from src.common.plot_utils import apply_plot_style
from src.common.config import ANALYSIS_OUTPUT_DIR

from src.analysis.scatter_plots import make_vre_storage_scatter, make_storage_solar_scatter
from src.analysis.boxplots import make_boxplot
from src.analysis.flex_metrics import compute_flex_ratios_all, make_analysis3_table
from src.analysis.flex_plots import make_figure4a, make_figure4b
from src.analysis.source_data import export_figure4_source_data
from src.analysis.stacked_bar import make_stacked_bar
from src.analysis.boxplot_indicators import make_analysis1_tables
from src.analysis.analysis_sensitivity import make_analysis2_sensitivity


def main(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    export_source_data: bool = True,
):
    apply_plot_style()
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Main manuscript figures
    fig3a, data3a = make_boxplot(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )
    
    analysis1_iam_spread, analysis1_benchmarks, analysis1_inset = make_analysis1_tables(
    save_csv=True,
)

    fig3b, data3b = make_vre_storage_scatter(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )
    
    fig_analysis2, data_analysis2 = make_analysis2_sensitivity(
    year=2050,
    plot_mode="range",
    save_png=save_png,
    save_pdf=save_pdf,
    save_svg=save_svg,
    save_csv=True,
)

    # Additional solar-share scatter with PyPSA benchmark and quantile trend
    fig_solar, data_solar = make_storage_solar_scatter(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
        add_pypsa=True,
        add_quantile_fit=True,
    )

    indices = compute_flex_ratios_all()

    fig4a, data4a = make_figure4a(
        indices=indices,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig4b, data4b = make_figure4b(
        indices=indices,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig_stack, data_stack = make_stacked_bar(
        years=(2030, 2050),
        as_share=False,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    analysis3_df, analysis3_corr = make_analysis3_table(
        save_csv=True,
        include_benchmarks=False,  # IAM-only main diagnostic
    )

    if export_source_data:
        export_figure4_source_data(
            figure3a=data3a,
            figure3b=data3b,
            figure4a=data4a,
            figure4b=data4b,
        )

    return {
        "Figure-3a": fig3a,
        "Figure-3b": fig3b,
        "Storage-solar-scatter": fig_solar,
        "Figure-4a": fig4a,
        "Figure-4b": fig4b,
        "Stacked-bar": fig_stack,
        "Analysis-1-IAM-spread": analysis1_iam_spread,
        "Analysis-1-benchmarks": analysis1_benchmarks,
        "Analysis-1-inset": analysis1_inset,
        "Analysis-2-sensitivity": fig_analysis2,
    }


if __name__ == "__main__":
    main()