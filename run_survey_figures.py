from src.common.config import SURVEY_XLSX, SURVEY_OUTPUT_DIR
from src.common.plot_utils import apply_plot_style

from src.survey.figure_2 import make_figure2
from src.survey.figure_3a import make_figure3a
from src.survey.figure_3b import make_figure3b
from src.survey.figure_s1 import make_figure_s1


def main():
    apply_plot_style()

    fig_2, _ = make_figure2(SURVEY_XLSX)
    fig_3a = make_figure3a(SURVEY_XLSX)
    fig_3b = make_figure3b(SURVEY_XLSX)
    fig_s1 = make_figure_s1(SURVEY_XLSX)
    
    # Output folder
    SURVEY_OUTPUT_DIR.mkdir(exist_ok=True)
    
    fig_2.savefig(SURVEY_OUTPUT_DIR / "Figure-2.png", dpi=300, bbox_inches="tight")
    fig_3a.savefig(SURVEY_OUTPUT_DIR / "Figure-3a.png", dpi=300, bbox_inches="tight")
    fig_3b.savefig(SURVEY_OUTPUT_DIR / "Figure-3b.png", dpi=300, bbox_inches="tight")
    fig_s1.savefig(SURVEY_OUTPUT_DIR / "Figure-S1.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

