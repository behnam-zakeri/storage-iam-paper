from __future__ import annotations

import seaborn as sns
from matplotlib import pyplot as plt

from src.common.config import (
    SCENARIOS_DIAGNOSTIC,
    SCENARIO_COLORS,
    SELECTED_REGION,
)
from src.common.plot_utils import save_figure
from src.analysis.data_loader import load_analysis_data
from src.supplementary.supplement_utils import compute_relative_change_vs_reference


def make_figure_s5(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    output_dir=None,
    selected_region: str = SELECTED_REGION,
    scenarios: list[str] | None = None,
    year: int = 2050,
    reference_scenario: str = "NetZero",
    exclude_models: list[str] | None = None,
):
    """
    Figure S5:
    Compare alternative scenarios of each model against its NetZero value.

    Returns
    -------
    fig : matplotlib.figure.Figure
    df_plot : pandas.DataFrame
        Underlying plotted data.
    """
    if scenarios is None:
        scenarios = SCENARIOS_DIAGNOSTIC
    if exclude_models is None:
        exclude_models = ["MEESA"]

    eu, eu_nzero, storage_var = load_analysis_data()

    iam_models = [x for x in eu.model if x not in exclude_models]

    df = (
        eu_nzero.filter(
            region=selected_region,
            scenario=scenarios,
            model=iam_models,
            year=year,
        )
        .timeseries()
        .reset_index()
    )

    df_plot = compute_relative_change_vs_reference(
        df,
        target_variable=storage_var,
        reference_scenario=reference_scenario,
        year=year,
        exclude_models=None,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")

    sns.barplot(
        data=df_plot,
        y="model",
        x="relative_change",
        hue="scenario",
        hue_order=scenarios,
        palette=SCENARIO_COLORS,
        dodge=True,
        ax=ax,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Relative change in capacity of storage (%)")
    ax.set_ylabel("Model")
    ax.set_title(f"Comparing all scenarios to NetZero (year={year})")
    ax.legend(
        title="Scenario",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()

    if output_dir is not None and (save_png or save_pdf or save_svg):
        save_figure(
            fig,
            output_dir / "Figure-S5",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

    return fig, df_plot

