from src.common.plot_utils import apply_plot_style
from src.common.config import ANALYSIS_OUTPUT_DIR

from src.analysis.scatter_plots import make_figure4b
from src.analysis.boxplots import make_figure4a
from src.analysis.flex_metrics import compute_flex_ratios_all
from src.analysis.flex_plots import make_figure4c
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
    fig4a, data4a = make_figure4a(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig4b, data4b = make_figure4b(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    indices = compute_flex_ratios_all()
    fig4c, data4c = make_figure4c(
        indices=indices,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    if export_source_data:
        export_figure4_source_data(
            figure4a=data4a,
            figure4b=data4b,
            figure4c=data4c,
        )

    return {
        "Figure-4a": fig4a,
        "Figure-4b": fig4b,
        "Figure-4c": fig4c,
    }


if __name__ == "__main__":
    main()