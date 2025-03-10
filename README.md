# COL_RES Multiomics Analysis Pipeline

## Overview

This repository provides a **Python-based pipeline** for integrating and analyzing **16S rRNA gene sequencing (microbiome) data** and **metabolomics data** in the context of the **Colonisation Resistance (COL_RES) ITN network**. The pipeline performs:

- **Data Ingestion & Preprocessing**: Merging 16S data at the genus level with metabolomic features.  
- **Differential Fold-Change Analysis**: Identifying up/downregulation relative to controls.  
- **Dimensionality Reduction (PCA)**: Extracting principal components to visualize group separations.  
- **Correlation Analyses**: Spearman (or Pearson) correlations between microbial genera and metabolites, with optional false-discovery rate correction.  
- **Hierarchical Clustering**: Grouping features with similar response profiles.  
- **Network Construction**: Building correlation-based bipartite networks to reveal key microbe-metabolite interactions.  

## Key Features

1. **Modular Pipeline** – Easily modify or remove steps without breaking the entire workflow.  
2. **High-Resolution Visualizations** – Automated generation of heatmaps, PCA plots, and network graphs.  
3. **Reproducible & Well-Documented** – The script includes comments and logs at each stage.  

## Repository Structure

```
COL_RES_Multiomics_Pipeline/
├── data/
│   ├── 16S_data.csv
│   ├── metabolomics_data.csv
├── scripts/
│   ├── main_pipeline.py
├── results/
│   ├── figures/
├── README.md
└── requirements.txt
```

- **data/**: Contains raw or preprocessed 16S and metabolomics data.  
- **scripts/**: Houses the main pipeline (`main_pipeline.py`) plus utility scripts for specific tasks (e.g., correlation analysis, plotting, etc.).  
- **results/**: Stores all output plots, tables, and logs generated by the pipeline.  
- **README.md**: This file.  
- **requirements.txt**: Python library dependencies.

## Installation

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/YourUsername/COL_RES_Multiomics_Pipeline.git
   cd COL_RES_Multiomics_Pipeline
   ```
2. **Create/activate a virtual environment** (recommended):  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Your Data**  
   - Ensure 16S data is aggregated at the genus level (e.g., in `16S_data.csv`).
   - Log-transform metabolomic intensities if desired, or let the script do it automatically.
   - Place all input files in the `data` folder.

2. **Configure the Script**  
   - Open `scripts/main_pipeline.py` and review the **user-defined parameters** (e.g., correlation threshold, reference control groups, etc.).
   - Update file paths or parameter settings to match your data.

3. **Run the Pipeline**  
   ```bash
   python scripts/main_pipeline.py
   ```
   - The script will load data, preprocess it, perform fold-change calculations, run PCA and correlation analysis, generate networks, and produce figures/tables.

4. **Check Outputs**  
   - All generated results (figures, tables, logs) will be saved in the `results/` folder by default.  
   - Customize or rename these outputs as needed for your own reports.

## Pipeline Steps in Brief

1. **Data Merging**: Combines 16S (genus-level) and metabolomic datasets by matching `[Mouse ID, Sample Day, Microbiome Type, Treatment Group]`.  
2. **Fold-Change Analysis**: Calculates log fold changes relative to control (PBS) for each feature.  
3. **PCA**: Reduces dimensionality to visualize sample clustering (pathogen vs. control, OMM12 vs. SPF).  
4. **Correlation**: Computes pairwise Spearman correlations between genera and metabolites; adjusts p-values for multiple comparisons.  
5. **Hierarchical Clustering**: Identifies clusters with similar expression/abundance shifts.  
6. **Network Construction**: Generates correlation-based bipartite graphs highlighting significant genus-metabolite interactions.
