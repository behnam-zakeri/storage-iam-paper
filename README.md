# Electricity Storage Representation Framework (SRF)

This repository provides the data and code supporting the analysis in:

Zakeri et al. (2026) — *Electricity storage in climate mitigation models* (under review).

## Scope

The repository contains three workflows:

1. **Survey analysis** – expert-based assessment of IAM storage representation (Figures 2, 3a-b, Figure S1) 
2. **Scenario analysis** – IAM scenario outputs and benchmarking (Figures 4a–c)  
3. **Supplementary analysis** – supplementary figures and robustness analysis (Figures S2–S16)

## Reproducibility

Main workflows can be reproduced by running:

    python run_survey_figures.py
    python run_analysis_figures.py
    python run_supplement_figures.py

Outputs are saved to:

    outputs/

## Data

Input datasets:

- IAM scenario data (IAMC format):  
  `data/analysis/ecemf_storage_in_iams.csv`
- PyPSA benchmark data:  
  `data/analysis/pypsa_capacities.xlsx`
- Survey data:  
  `data/survey/storage-in-IAMs_submission.xlsx`

Derived data:

- Figure 4 source data:  
  `data/analysis/Figure-4_source-data.xlsx`

## Methodological notes

### Storage intensity (peak load)

Peak electricity demand is inferred from annual electricity demand:

    GW_peak = EJ * 31.7 / load_factor

with:
- load factor = 0.62 (Europe)

### Hydrogen capacity inference

Where electrolyser capacity is not reported:

- efficiency = 0.70  
- capacity factor = 0.70  

Capacity is inferred from hydrogen production.

### Flexibility indicators

Indicators include:

- storage capacity ratios  
- flexible generation  
- demand-side flexibility proxy  
- hydrogen electrolysis  
- spatial flexibility  
- curtailment  

## Repository structure

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

## Workflows

### Survey workflow
Generates:
- Figure 2
- Figure 3a
- Figure 3b
- Figure S1

### Analysis workflow
Generates:
- Figure 4a
- Figure 4b
- Figure 4c

### Supplementary workflow
Generates:
- Figures S2–S9 individually
- Figures S10–S16 in a model-loop generator

## Notebooks

Interactive notebooks are provided for each workflow:

- `notebooks/make_survey_figures.ipynb`
- `notebooks/make_analysis_figures.ipynb`
- `notebooks/make_supplementary_figures.ipynb`

## Environment

Python 3.11

Required packages:
- pyam
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl

## Status

This repository reflects the analysis submitted for peer review.

Results and data may be updated following the review process.

## Contributors

Current repository lead:
- Behnam Zakeri (zakeri@iiasa.ac.at)

Additional contributors may be added as the project evolves.

## Citation and sharing

This project is not final. Please do not share the data and figures outside the review process.

## License

MIT License
