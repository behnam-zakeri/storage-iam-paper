from src.common.plot_utils import apply_plot_style
from src.common.config import SUPPLEMENTARY_OUTPUT_DIR

from src.supplementary.figure_s2 import make_figure_s2
from src.supplementary.figure_s3 import make_figure_s3
from src.supplementary.figure_s4 import make_figure_s4
from src.supplementary.figure_s5 import make_figure_s5
from src.supplementary.figure_s6 import make_figure_s6
from src.supplementary.figure_s7 import make_figure_s7
from src.supplementary.figures_s8_s9 import (make_storage_to_vre_scatter,
                                        make_storage_to_total_scatter)
from src.supplementary.figures_s10_s16 import make_figures_s10_s16
from src.supplementary.source_data import export_supplementary_source_data


def main(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    export_source_data: bool = True,
):
    apply_plot_style()
    SUPPLEMENTARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # S2-S4 currently return figures only
    fig_s2 = make_figure_s2(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig_s3 = make_figure_s3(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig_s4 = make_figure_s4(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    # S5-S9 return (figure, source_data)
    fig_s5, data_s5 = make_figure_s5(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
        output_dir=SUPPLEMENTARY_OUTPUT_DIR,
    )

    fig_s6, data_s6 = make_figure_s6(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
        output_dir=SUPPLEMENTARY_OUTPUT_DIR,
    )

    fig_s7, data_s7 = make_figure_s7(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
        output_dir=SUPPLEMENTARY_OUTPUT_DIR,
    )

    fig_s8, data_s8 = make_storage_to_vre_scatter(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    fig_s9, data_s9 = make_storage_to_total_scatter(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )
    figs_s10_s16 = make_figures_s10_s16(
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    if export_source_data:
        source_tables = {
            "Figure-S5": data_s5,
            "Figure-S6": data_s6,
            "Figure-S7": data_s7,
            "Figure-S8": data_s8,
            "Figure-S9": data_s9,
        }

        for fig_label, (_, df_plot) in figs_s10_s16.items():
            source_tables[fig_label] = df_plot

        export_supplementary_source_data(source_tables)

    return {
        "Figure-S2": fig_s2,
        "Figure-S3": fig_s3,
        "Figure-S4": fig_s4,
        "Figure-S5": (fig_s5, data_s5),
        "Figure-S6": (fig_s6, data_s6),
        "Figure-S7": (fig_s7, data_s7),
        "Figure-S8": (fig_s8, data_s8),
        "Figure-S9": (fig_s9, data_s9),
        **figs_s10_s16,
    }


if __name__ == "__main__":
    main()