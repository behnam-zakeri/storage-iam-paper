from src.common.config import SURVEY_XLSX, SURVEY_OUTPUT_DIR
from src.common.plot_utils import apply_plot_style

from src.survey.figure_2a import make_figure2a
from src.survey.figure_2b import make_figure2b
from src.survey.figure_s1 import make_figure_s1
from src.survey.figure_s2 import make_figure_s2


def main():
    apply_plot_style()

    fig_2a = make_figure2a(SURVEY_XLSX)
    fig_2b = make_figure2b(SURVEY_XLSX)
    fig_s1 = make_figure_s1(SURVEY_XLSX)
    fig_s2, _ = make_figure_s2(SURVEY_XLSX)
    
    # Output folder
    SURVEY_OUTPUT_DIR.mkdir(exist_ok=True)
    
    fig_2a.savefig(SURVEY_OUTPUT_DIR / "Figure-2a.png", dpi=300, bbox_inches="tight")
    fig_2b.savefig(SURVEY_OUTPUT_DIR / "Figure-2b.png", dpi=300, bbox_inches="tight")
    fig_s1.savefig(SURVEY_OUTPUT_DIR / "Figure-S1.png", dpi=300, bbox_inches="tight")
    fig_s2.savefig(SURVEY_OUTPUT_DIR / "Figure-S2.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

