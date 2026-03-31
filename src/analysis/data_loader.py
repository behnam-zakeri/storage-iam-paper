from __future__ import annotations

import pyam
from pyam import IamDataFrame

from src.common.config import (
    IAMC_CSV,
    REGIONS_EU,
    MODEL_EXCLUDE,
    RENAME_MODEL,
    RENAME_SCENARIO,
    SCENARIOS_DIAGNOSTIC,
    COLOR_MODEL,
    COLOR_CODE,
    MARKER_LIST,
    PUMPED_HYDRO_INCLUDED,
    VAR_STORAGE_POWER,
)


def apply_pyam_run_control() -> None:
    """
    Apply shared pyam plotting colours.

    This is safe to call multiple times.
    """
    pyam.run_control().update({"color": {"model": COLOR_MODEL}})
    pyam.run_control().update({"color": {"variable": COLOR_CODE}})


def load_raw_iamc_data() -> IamDataFrame:
    """
    Load the raw IAMC-format CSV as a pyam IamDataFrame.
    """
    apply_pyam_run_control()
    return IamDataFrame(data=IAMC_CSV)


def _subtract_pumped_hydro(eu: IamDataFrame) -> tuple[IamDataFrame, str]:
    """
    Create a non-pumped-hydro storage-power variable.

    Returns
    -------
    eu : IamDataFrame
        Updated dataframe including a derived storage variable without pumped hydro.
    storage_var : str
        Variable name to use downstream.
    """
    ph_var = "Capacity|Electricity|Storage|Power|Pumped Hydro"
    new_var = "Capacity|Electricity|Storage|Power_new"

    # models that explicitly report pumped hydro
    models_with_ph = eu.filter(variable=ph_var).model
    models_wo_ph = [x for x in eu.model if x not in models_with_ph]

    # total minus pumped hydro
    df_ph = eu.filter(model=models_with_ph, variable=[VAR_STORAGE_POWER, ph_var])
    if not df_ph.empty:
        df_ph.subtract(VAR_STORAGE_POWER, ph_var, new_var, append=True)
        eu = eu.append(df_ph.filter(variable=new_var))

    # for models without pumped hydro, duplicate total storage as new_var
    if models_wo_ph:
        df_no_ph = eu.filter(model=models_wo_ph, variable=VAR_STORAGE_POWER).rename(
            variable={VAR_STORAGE_POWER: new_var}
        )
        eu = eu.append(df_no_ph.filter(variable=new_var))

    return eu, new_var


def load_analysis_data() -> tuple[IamDataFrame, IamDataFrame, str]:
    """
    Load, harmonise, and filter the IAMC analysis dataset.

    Steps
    -----
    1. Load raw IAMC CSV
    2. Exclude unwanted model versions
    3. Filter to Europe
    4. Rename models and scenarios
    5. Keep only harmonised scenarios
    6. Optionally exclude pumped hydro from storage power
    7. Build the net-zero subset

    Returns
    -------
    eu : IamDataFrame
        Harmonised Europe dataset.
    eu_nzero : IamDataFrame
        Net-zero scenario subset.
    storage_var : str
        Storage-power variable to use downstream.
    """
    data = load_raw_iamc_data()

    # exclude selected model versions
    data = data.filter(model=MODEL_EXCLUDE, keep=False)

    # Europe only
    eu = data.filter(region=REGIONS_EU)

    # harmonise names
    eu = eu.rename(model=RENAME_MODEL)
    eu = eu.rename(scenario=RENAME_SCENARIO)

    # keep only scenarios explicitly defined in the harmonised mapping
    eu = eu.filter(scenario=list(RENAME_SCENARIO.values()))

    # storage variable to use downstream
    storage_var = VAR_STORAGE_POWER

    if not PUMPED_HYDRO_INCLUDED:
        eu, storage_var = _subtract_pumped_hydro(eu)

    # net-zero subset
    eu_nzero = eu.filter(scenario="*NetZero*")

    return eu, eu_nzero, storage_var


def get_reporting_storage_models(
    eu: IamDataFrame,
    storage_var: str,
) -> list[str]:
    """
    Return models that report the requested storage variable.
    """
    return list(eu.filter(variable=storage_var).model)


def filter_to_storage_reporting_models(
    eu: IamDataFrame,
    storage_var: str,
) -> IamDataFrame:
    """
    Keep only models that report the requested storage variable.
    """
    models = get_reporting_storage_models(eu, storage_var)
    return eu.filter(model=models)


def build_model_markers(idf: IamDataFrame) -> dict[str, str]:
    """
    Assign a stable marker to each model appearing in the supplied dataframe.
    """
    models = list(idf.model)
    return {
        model: MARKER_LIST[i % len(MARKER_LIST)]
        for i, model in enumerate(models)
    }


def get_pypsa_storage_row_label() -> str:
    """
    Return the expected PyPSA storage row label depending on whether
    pumped hydro is included in the storage definition.
    """
    if PUMPED_HYDRO_INCLUDED:
        return "Electricity storage"
    return "battery discharger"