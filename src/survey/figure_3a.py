import matplotlib.pyplot as plt

from src.common.config import (
    SHEET_FIG3,
    MODEL_ORDER,
    MODEL_DISPLAY,
    DIMENSION_MAP,
    DIMENSION_SPECS,
)
from src.common.io_utils import read_excel_sheet, clean_text
from src.common.plot_utils import lighten_color
from src.survey.srf_utils import (
    build_long_srf,
    attach_criteria_order,
    draw_segmented_glyph,
    build_guide_df,
)


def make_figure3a(xlsx_file):
    df_raw = read_excel_sheet(xlsx_file, SHEET_FIG3)

    dimension_col = df_raw.columns[0]
    criterion_col = df_raw.columns[1]
    max_col = df_raw.columns[-1]

    # clean headers/merged cells
    df_raw[dimension_col] = df_raw[dimension_col].ffill().apply(clean_text)
    df_raw[criterion_col] = df_raw[criterion_col].apply(clean_text)

    # harmonise dimension names
    df_raw[dimension_col] = df_raw[dimension_col].replace(DIMENSION_MAP)

    # model columns
    model_cols = list(df_raw.columns[2:-1])
    model_cols = [m for m in MODEL_ORDER if m in model_cols]

    # drop subtotal rows for panel a
    df_use = df_raw[
        df_raw[criterion_col].astype(str).str.strip().str.lower() != "subtotal"
    ].copy()

    # long form + norms
    df = build_long_srf(df_use, dimension_col, criterion_col, max_col, model_cols)

    # attach criterion order to specs
    specs = attach_criteria_order(df_use, dimension_col, criterion_col, DIMENSION_SPECS)

    def add_explainer(ax):
        guide_df = build_guide_df(specs)
        draw_segmented_glyph(
            ax,
            guide_df,
            specs,
            lighten_color=lighten_color,
            show_dim_labels_guide=True,
            show_criterion_labels=True,
        )
        ax.set_title("Guidance", fontsize=11, pad=15, color="0.15", fontweight="bold")

    # layout
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1, 1, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.02,
        hspace=0.06
    )

    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]

    for pos, model in zip(positions, model_cols):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        sub = (
            df[df["model"] == model][[dimension_col, criterion_col, "norm"]]
            .rename(columns={dimension_col: "dimension", criterion_col: "criterion"})
        )

        draw_segmented_glyph(
            ax,
            sub,
            specs,
            lighten_color=lighten_color,
            show_dim_labels=True,
            show_criterion_labels=False
        )
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11, pad=1)

    guide_ax = fig.add_subplot(gs[:, 4])
    add_explainer(guide_ax)

    fig.suptitle(
        "a) Model fingerprints across four storage representation dimensions",
        y=0.97, fontsize=13.5, fontweight="bold"
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.99, 0.94])
    return fig