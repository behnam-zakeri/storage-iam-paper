from __future__ import annotations

from src.common.config import (
    MODEL_ORDER,
    SELECTED_REGION,
    SUPPLEMENTARY_OUTPUT_DIR,
)
from src.common.plot_utils import save_figure
from src.supplementary.supplement_utils import (
    get_storage_reporting_subset,
    add_vre_generation_and_share,
)
from src.supplementary.model_scatter_utils import (
    build_model_yearly_scatter_tables,
    plot_single_model_scenario_scatter,
)


def make_figures_s10_s16(
    save_png: bool = True,
    save_pdf: bool = False,
    save_svg: bool = False,
    selected_region: str = SELECTED_REGION,
    years=range(2020, 2051, 5),
    exclude_models: list[str] | None = None,
):
    """
    Generate Figures S10-S16:
    capacity of storage vs share of VRE, one panel per model.

    The figure numbering follows MODEL_ORDER after filtering to models that
    report storage. This keeps numbering stable across runs.

    Returns
    -------
    results : dict[str, tuple[Figure, DataFrame]]
        Mapping from figure label to (figure, plotted data).
    """
    if exclude_models is None:
        exclude_models = ["Euro-Calliope"]

    df, eu_storage, storage_var = get_storage_reporting_subset(
        selected_region=selected_region,
        years=years,
    )

    df = add_vre_generation_and_share(df)

    var_x = "Share in Secondary Energy|Electricity|VRE"
    var_y = storage_var

    reporting_models = list(eu_storage.model)
    models_final = sorted(
        [m for m in reporting_models if m not in exclude_models]
    )

    results = {}

    for i, model in enumerate(models_final, start=10):
        figure_label = f"Figure-S{i}"

        df_sub = df.filter(model=model, variable=[var_x, var_y])
        df_plot = build_model_yearly_scatter_tables(
            df_sub,
            var_x=var_x,
            var_y=var_y,
            years=years,
        )

        if df_plot.empty:
            continue

        fig, ax = plot_single_model_scenario_scatter(
            df_plot,
            model=model,
            var_x=var_x,
            var_y=var_y,
            xlabel="Share of VRE from total electricity generation",
            ylabel="Capacity of storage (GW)",
            title=f"Capacity of storage relative to share of VRE (model={model})",
            xlim=(0, 1),
            ylim=None,
            marker_size=100,
            legend_frameon=False,
        )

        save_figure(
            fig,
            SUPPLEMENTARY_OUTPUT_DIR / f"{figure_label}_{model}",
            save_png=save_png,
            save_pdf=save_pdf,
            save_svg=save_svg,
        )

        results[figure_label] = (fig, df_plot)

    return results

