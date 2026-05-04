from src.common.plot_utils import apply_plot_style
from src.common.config import ANALYSIS_OUTPUT_DIR

from src.analysis.scatter_plots import make_figure3b
from src.analysis.boxplots import make_figure3a
from src.analysis.flex_metrics import compute_flex_ratios_all
from src.analysis.flex_plots import make_figure4a, make_figure4b
from src.analysis.source_data import export_figure4_source_data


def main(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    export_source_data: bool = True,
):
    apply_plot_style()
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Main manuscript figures
    fig3a, data3a = make_figure3a(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig3b, data3b = make_figure3b(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
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
        "Figure-4a": fig4a,
        "Figure-4b": fig4b,
    }


if __name__ == "__main__":
    main()