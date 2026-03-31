from pathlib import Path

# =========================================================
# Project root
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# Shared folders
# =========================================================
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

# =========================================================
# Survey paths
# =========================================================
SURVEY_DATA_DIR = DATA_DIR / "survey"
SURVEY_OUTPUT_DIR = OUTPUT_DIR / "survey"

SURVEY_XLSX = SURVEY_DATA_DIR / "storage-in-IAMs_submission.xlsx"

# Survey sheet names
SHEET_FIG2 = "Figure-2"
SHEET_FIG3 = "Figure-3"
SHEET_S1 = "SRF-total"

# =========================================================
# Analysis paths
# =========================================================
ANALYSIS_DATA_DIR = DATA_DIR / "analysis"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"

# =========================================================
# Supplementary paths
# =========================================================

SUPPLEMENTARY_DATA_DIR = DATA_DIR / "supplementary"
SUPPLEMENTARY_OUTPUT_DIR = OUTPUT_DIR / "supplementary"

IAMC_CSV = ANALYSIS_DATA_DIR / "ecemf_storage_in_iams.csv"
PYPSA_XLSX = ANALYSIS_DATA_DIR / "pypsa_capacities.xlsx"
PYPSA_SHEET = "pypsa"
FIG4_SOURCE_DATA_XLSX = ANALYSIS_DATA_DIR / "Figure-4_source-data.xlsx"

# =========================================================
# Shared model names
# =========================================================
MODEL_ORDER = [
    "AIM-T",
    "GCAM",
    "IMAGE",
    "MESSAGEix",
    "PROMETHEUS",
    "REMIND",
    "TIAM-ECN",
    "WITCH",
]

MODEL_DISPLAY = {
    "AIM-T": "AIM-T",
    "GCAM": "GCAM",
    "IMAGE": "IMAGE",
    "MESSAGEix": "MESSAGEix",
    "PROMETHEUS": "PROMETHEUS",
    "REMIND": "REMIND",
    "TIAM-ECN": "TIAM-ECN",
    "WITCH": "WITCH",
}

MODEL_DISPLAY_SHORT = {
    "AIM-T": "AIM-T",
    "GCAM": "GCAM",
    "IMAGE": "IMAGE",
    "MESSAGEix": "MESSAGEix",
    "PROMETHEUS": "PROM.",
    "REMIND": "REMIND",
    "TIAM-ECN": "TIAM-ECN",
    "WITCH": "WITCH",
}

# =========================================================
# Survey harmonisation
# =========================================================
CATEGORY_MAP = {
    "techno-economic": "Techno-economic",
    "technoeconomic": "Techno-economic",
    "techno economic": "Techno-economic",
    "applications and services": "Applications and services",
    "integration constraints": "Integration constraints",
}

CATEGORY_ORDER = [
    "Techno-economic",
    "Applications and services",
    "Integration constraints",
]

DIMENSION_MAP = {
    "Explicit storage representation": "Storage representation",
    "Breadth in VRE modelling": "VRE integration",
    "Spatial structure": "Spatial representation",
    "Temporal representation": "Temporal representation",
    # allow already-clean names too
    "Storage representation": "Storage representation",
    "VRE integration": "VRE integration",
    "Spatial representation": "Spatial representation",
}

DIMENSION_SPECS = {
    "Storage representation": {
        "theta1": 45,
        "theta2": 135,
        "color": "#1b9e77",
        "short": "Storage",
    },
    "Spatial representation": {
        "theta1": 135,
        "theta2": 225,
        "color": "#4c78a8",
        "short": "Spatial",
    },
    "Temporal representation": {
        "theta1": 225,
        "theta2": 315,
        "color": "#7570b3",
        "short": "Temporal",
    },
    "VRE integration": {
        "theta1": 315,
        "theta2": 405,
        "color": "#d95f02",
        "short": "VRE",
    },
}

# =========================================================
# Survey figure constants
# =========================================================
MAX_MODELS = 8
PURPLE = "mediumorchid"

FIG2_STEM_LW = 2.2
FIG2_DOT_S = 70

FIG3A_LAYOUT = {
    "r_max": 1.18,
    "r_mid": 0.59,
    "inner_hole": 0.08,
    "sep_deg": 1.8,
}

S1_SEGMENTS = 6

# =========================================================
# Analysis settings
# =========================================================
PUMPED_HYDRO_INCLUDED = False
CORRECT_WITCH = True
APPEND_AIM = True

REGIONS_EU = ["EU27 & UK (*)", "EU27 (*)"]
SELECTED_REGION = "EU27 & UK (*)"
SELECTED_SCENARIO = "NetZero"

MODEL_EXCLUDE = [
    "Enertile 6.50.0",
    "Euro-Calliope 2.0",
    "IMAGE 3.2",
    "IMAGE 3.3",
    "LIMES 2.38",
    "OSeMBE v1.0.0",
    "PRIMES 2022",
    "REMIND 2.1",
    "REMIND 3.0.0",
    "TIAM-ECN 1.2",
    "WITCH 5.0",
    "WITCH 5.1",
]

RENAME_MODEL = {
    "AIM/Technology 2.0": "AIM-T",
    "IMAGE 3.4": "IMAGE",
    "MEESA v1.1": "MEESA",
    "MESSAGEix-GLOBIOM 1.2": "MESSAGEix",
    "PROMETHEUS 1.2": "PROMETHEUS",
    "REMIND 3.3": "REMIND",
    "TIAM-ECN 1.3": "TIAM-ECN",
    "WITCH 5.4": "WITCH",
}

RENAME_SCENARIO = {
    "WP1 NPI": "NPI",
    "WP1 NetZero": "NetZero",
    "WP1 NetZero-HighEfficiency": "NetZero|High-eff",
    "WP1 NetZero-LimBio": "NetZero|LimBio",
    "WP1 NetZero-LimCCS": "NetZero|LimCCS",
    "WP1 NetZero-LimNuc": "NetZero|LimNuc",
    "WP1 NetZero-SynfPush": "NetZero|e-fuel",
}

SCENARIO_COLORS = {
    "NPI": "blue",
    "NetZero": "orchid",
    "NetZero|High-eff": "salmon",
    "NetZero|LimBio": "mediumaquamarine",
    "NetZero|LimCCS": "steelblue",
    "NetZero|LimNuc": "purple",
    "NetZero|e-fuel": "#60c4ac",
}

SCENARIOS_DIAGNOSTIC = [
    "NetZero",
    "NetZero|LimBio",
    "NetZero|LimCCS",
    "NetZero|LimNuc",
]

COLOR_MODEL = {
    "AIM-T": "black",
    "IMAGE": "yellow",
    "MEESA": "orange",
    "MESSAGEix": "violet",
    "PROMETHEUS": "blue",
    "REMIND": "cyan",
    "TIAM-ECN": "c",
    "WITCH": "limegreen",
}

COLOR_VAR = {
    "Electricity": "darkblue",
    "Gases": "brown",
    "Hydrogen": "purple",
    "Liquids": "rosybrown",
}

COLOR_CODE = {
    "Electricity": "crimson",
    "Gases": "cyan",
    "Gases|Electricity": "c",
    "Geothermal": "olive",
    "Heat": "slateblue",
    "Solar": "orange",
    "Other": "mediumseagreen",
    "Hydrogen": "mediumorchid",
    "Liquids": "rosybrown",
    "Liquids|Electricity": "indianred",
    "Solids": "darkcyan",
    "Waste": "brown",
}

FUEL_LIST = ["Electricity", "Fossil", "Hydrogen", "Biomass", "Solar"]
MARKER_LIST = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

# Main IAMC variables
VAR_STORAGE_POWER = "Capacity|Electricity|Storage|Power"
VAR_ELEC_CAPACITY = "Capacity|Electricity"

VAR_ELEC_TOTAL = "Secondary Energy|Electricity"
VAR_ELEC_WIND = "Secondary Energy|Electricity|Wind"
VAR_ELEC_SOLAR = "Secondary Energy|Electricity|Solar"
VAR_ELEC_VRE = "Secondary Energy|Electricity|VRE"

VAR_ELEC_GAS = "Secondary Energy|Electricity|Gas"
VAR_ELEC_HYDRO = "Secondary Energy|Electricity|Hydro"
VAR_ELEC_BIOMASS = "Secondary Energy|Electricity|Biomass"
VAR_ELEC_CURT = "Secondary Energy|Electricity|Curtailment"

VAR_CAP_WIND = "Capacity|Electricity|Wind"
VAR_CAP_SOLAR = "Capacity|Electricity|Solar"
VAR_CAP_VRE = "Capacity|Electricity|VRE"

VAR_FE_ELEC = "Final Energy|Electricity"
VAR_FE_TOTAL = "Final Energy"
VAR_FE_ELEC_PEAK = "Final Energy|Electricity|Peak"

VAR_NET_TRADE = "Trade|Secondary Energy|Electricity|Volume"

VAR_H2_FROM_ELEC = "Secondary Energy|Hydrogen|Electricity"
VAR_H2_ELEC_CAP = "Capacity|Hydrogen|Electricity"

VAR_STORAGE_REL_TOTAL = "Relative Capacity|Electricity|Storage to Total"
VAR_STORAGE_REL_VRE = "Relative Capacity|Electricity|Storage to VRE"
VAR_VRE_SHARE = "Share in Secondary Energy|Electricity|VRE"
VAR_STORAGE_REL_PEAK = "Relative Final Energy|Electricity|Storage to Peak"

# Analysis numeric assumptions
LOAD_FACTOR_ANALYSIS = 0.64
LOAD_FACTOR_EU = 0.62
EJYR_TO_GWAVG = 31.7
DEMAND_FLEX_SHARE = 0.20
H2_ELECTROLYZER_EFF = 0.70
H2_CAPACITY_FACTOR = 0.70
EJ_PER_GWYR = 0.031536

# Spatial annotations / assumptions
EUUK_N_REGIONS = 28

SPATIAL_RESOLUTION_MAP = {
    "AIM-T": "Low",
    "MESSAGEix": "Low",
    "MEESA": "Medium",
    "REMIND": "Medium",
    "IMAGE": "Low",
    "PROMETHEUS": "Low",
    "TIAM-ECN": "Low",
    "WITCH": "Very low",
}

SPATIAL_RES_TO_NREG = {
    "Very low": 1,
    "Low": 2,
    "Medium": 5,
}

DR_TIER_MAP = {
    "AIM-T": "High",
    "MESSAGEix": "High",
    "REMIND": "No",
    "IMAGE": "No",
    "MEESA": "High",
    "PROMETHEUS": "No",
    "TIAM-ECN": "No",
    "WITCH": "No",
}

# Figure 4 defaults
FIG4_YEARS = [2030, 2040, 2050]
FIG4_SCEN_NZ = "NetZero"
FIG4_SCEN_DIAG = "NetZero|LimCCS"
FIG4_SCEN_FALLBACK = "NetZero|LimBio"

ENTSOE_POLICY_DATA = {
    2030: 56,
    2040: 227,
    2050: 540,
}

EU_INSTALLED_2025_STORAGE_GW = 48