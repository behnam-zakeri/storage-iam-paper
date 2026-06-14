# Electricity Storage Representation Framework (SRF)

This repository provides the data and code supporting the analysis in:

Zakeri et al. (2026): *Interpreting electricity storage in net-zero energy pathways* (under review).

## Scope

The repository contains three main workflows:

1. **Survey analysis** – expert-based assessment of electricity storage representation in climate mitigation models.
2. **Scenario analysis** – IAM scenario outputs, benchmark comparisons, storage–renewables diagnostics, and flexibility portfolio analysis.
3. **Supplementary analysis** – supplementary figures, robustness checks, and model-specific diagnostics.

The scenario analysis workflow now also includes additional quantitative diagnostics used to interpret the main figures, including IAM spread indicators, benchmark comparisons, within-model scenario sensitivity, and a solar-share scatter diagnostic with PyPSA-Eur benchmark and optional median quantile-regression trend.

## Reproducibility

Main workflows can be reproduced by running:

```bash
python run_survey_figures.py
python run_analysis_figures.py
python run_supplement_figures.py
```

Outputs are saved to:

```text
outputs/
```

Main analysis outputs are saved to:

```text
outputs/analysis/
```

## Data

Input datasets:

* IAM scenario data in IAMC format:
  `data/analysis/ecemf_netzero_scenario_data.csv`

* PyPSA-Eur benchmark data:
  `data/analysis/pypsa_eur_data.xlsx`

* Survey data:
  `data/survey/storage-in-IAMs_survey.xlsx`

Derived/source data:

* Main analysis source data and diagnostic tables:
  `outputs/analysis/`

* Supplementary figures source data:
  `data/supplementary/Figure-S_source-data.xlsx`

## Methodological notes

### Storage intensity relative to peak load

Storage intensity is defined as electricity storage power capacity divided by inferred peak electricity demand:

```text
storage intensity = GW_storage / GW_peak
```

Peak electricity demand is inferred from annual final electricity demand:

```text
GW_peak = EJ_electricity × 31.7 / load_factor
```

with:

* `31.7` converting EJ/yr to GW average load;
* load factor defined in `src/common/config.py`.

This metric is used in the storage–VRE and storage–solar scatter diagnostics.

### VRE and solar shares

The storage scatter diagnostics use two related x-axis metrics:

* VRE share in electricity generation:

```text
(wind + solar generation) / total electricity generation
```

* Solar share in electricity generation:

```text
solar generation / total electricity generation
```

The VRE-share scatter is used for the main storage–renewables diagnostic. The solar-share scatter is an additional analysis figure with benchmark and trend overlays.

### PyPSA-Eur benchmark treatment

PyPSA-Eur benchmark results are read from:

```text
data/analysis/pypsa_eur_data.xlsx
```

For scatter plots, PyPSA-Eur is shown as a weather-year benchmark distribution using:

* a median marker;
* a shaded ellipse summarising the distribution across weather years.

The PyPSA-Eur points are saved as CSV source data when the relevant figure functions are run.

### Median quantile-regression trend

The solar-share scatter can optionally include a dashed median quantile-regression trend. This requires:

```text
statsmodels
```

If `statsmodels` is not installed, run the solar scatter with:

```python
add_quantile_fit=False
```

or install:

```bash
pip install statsmodels
```

### Hydrogen capacity inference

Where electrolyser capacity is not reported, capacity is inferred from hydrogen production using assumptions defined in `src/common/config.py`, including:

* electrolyser efficiency;
* electrolyser capacity factor;
* EJ-to-GW-year conversion.

### Flexibility indicators

Flexibility portfolio indicators include:

* electricity storage capacity ratio;
* flexible generation;
* demand-side flexibility proxy;
* hydrogen electrolysis;
* spatial flexibility;
* curtailment.

These indicators support the flexibility portfolio figures and diagnostics.

## Repository structure

```text
storage-srf/
  data/
    survey/
    analysis/
    supplementary/
  outputs/
    survey/
    analysis/
    supplementary/
  src/
    common/
    survey/
    analysis/
    supplementary/
  notebooks/
```

## Workflows

### Survey workflow

Run:

```bash
python run_survey_figures.py
```

Generates survey-based SRF figures, including:

* segmented model glyphs;
* criterion-level heatmaps;
* supplementary survey diagnostics.

Relevant modules include:

```text
src/survey/
```

### Scenario analysis workflow

Run:

```bash
python run_analysis_figures.py
```

Generates the main scenario-analysis figures and diagnostic tables.

Current outputs include:

* storage capacity boxplots;
* storage intensity versus VRE share scatter;
* storage intensity versus solar share scatter;
* stacked power-capacity bars;
* flexibility portfolio plots;
* flexibility portfolio diagnostics;
* IAM spread and benchmark comparison tables;
* within-model scenario-sensitivity diagnostics.

Relevant modules include:

```text
src/analysis/boxplots.py
src/analysis/scatter_plots.py
src/analysis/stacked_bar.py
src/analysis/flex_metrics.py
src/analysis/flex_plots.py
src/analysis/analysis1_indicators.py
src/analysis/analysis2_sensitivity.py
```

### Additional analysis modules

#### Analysis 1: IAM spread and benchmark indicators

Module:

```text
src/analysis/analysis1_indicators.py
```

Callable function:

```python
make_analysis1_tables()
```

Outputs:

```text
outputs/analysis/analysis1_iam_spread.csv
outputs/analysis/analysis1_benchmarks_vs_iam.csv
outputs/analysis/analysis1_inset_table_netzero.csv
```

This analysis summarises IAM spread by scenario and year, compares benchmark and policy estimates against IAM distributions, and produces a compact NetZero inset table.

#### Analysis 2: Within-model scenario sensitivity

Module:

```text
src/analysis/analysis2_sensitivity.py
```

Callable function:

```python
make_analysis2_sensitivity()
```

Outputs:

```text
outputs/analysis/analysis2_storage_intensity_2050.csv
outputs/analysis/analysis2_storage_scenario_deltas_2050.csv
outputs/analysis/analysis2_storage_diagnostic_deltas_2050.csv
outputs/analysis/analysis2_sensitivity_table_2050.csv
outputs/analysis/analysis2_storage_scenario_sensitivity_2050_range.png
```

This analysis quantifies within-model changes in electricity storage intensity across NetZero and diagnostic scenarios, including LimBio, LimCCS, and LimNuc.

#### Solar-share scatter diagnostic

Module:

```text
src/analysis/scatter_plots.py
```

Callable function:

```python
make_storage_solar_scatter()
```

Outputs:

```text
outputs/analysis/Figure-storage-solar-scatter.png
outputs/analysis/Figure-storage-solar-scatter_source_data.csv
outputs/analysis/Figure-storage-solar-scatter_pypsa_points.csv
outputs/analysis/Figure-storage-solar-scatter_qreg_fit.csv
outputs/analysis/Figure-storage-solar-scatter_qreg_summary.csv
outputs/analysis/Figure-storage-solar-scatter_binned_trend.csv
```

The quantile-regression outputs are generated only when `add_quantile_fit=True` and `statsmodels` is installed.

### Supplementary workflow

Run:

```bash
python run_supplement_figures.py
```

Generates supplementary figures and robustness diagnostics.

Relevant modules include:

```text
src/supplementary/
```

## Main analysis interface

The main analysis workflow is controlled by:

```text
run_analysis_figures.py
```

It calls the main figure functions and supporting analysis modules, including:

```python
make_boxplot()
make_storage_vre_scatter()
make_storage_solar_scatter()
make_stacked_bar()
compute_flex_ratios_all()
make_figure4a()
make_figure4b()
make_analysis1_tables()
make_analysis2_sensitivity()
make_analysis3_table()
```

The script also exports source data where configured.

## Notebooks

Interactive notebooks are provided for each workflow:

```text
notebooks/make_survey_figures.ipynb
notebooks/make_analysis_figures.ipynb
notebooks/make_supplementary_figures.ipynb
```

## Environment

Python 3.11 is recommended.

Required packages:

* pyam
* pandas
* numpy
* matplotlib
* seaborn
* openpyxl

Optional package:

* statsmodels, required only for the median quantile-regression trend in the solar-share scatter diagnostic.

Install optional dependency with:

```bash
pip install statsmodels
```

## Status

This repository reflects the analysis prepared for peer review.

Results, data, figure numbering, and file names may be updated during the review process.

## Contributors

Current repository lead:

* Behnam Zakeri
  [zakeri@iiasa.ac.at](mailto:zakeri@iiasa.ac.at)

## Citation and sharing

This project is not final. Please do not share the data and figures outside the review process.

## License

MIT License
