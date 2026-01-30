# Multimodal Atlas Building and Query prediction using Orchestrated Approximate Message Passing

This GitHub repository contains codes for implementing the techniques for building multimodal cell atlas and querying new subjects that are only partially observed via the algorithms proposed in [Multimodal data integration and cross-modal querying via orchestrated approximate message passing](https://arxiv.org/abs/2407.19030) by Nandy and Ma (2024). 

## Installation Guide

This repository contains three parts: simulation, analysis of TEA-seq data and analysis of CITE-seq data. The simulation part is based on python scripts that were executed on ASC Unity cluster of Ohio State University. 

### Setting up python environment for simulation

1. **Install Anaconda or Miniconda**:
   If you haven't already, download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Virtual Environment**:
  1. From the root of the repository, run:
    ```bash
     python -m venv orchamp-simulation

3. **Activate the Virtual Environment**:
   Once the environment is created, activate it using the following command:
   ```bash
   conda activate orchamp-simulation

4. **Install required packages**:
   You can install the required packages using
   ```bash
   pip install numpy scipy matplotlib pandas scikit-learn seaborn tqdm scanpy
   
5. **R Installation and rpy2**:
   You might need to install `R` in the virtual environment and install the package `rpy2`.


### Setting up Python environment for data analysis

The data analysis is based on Jupyter notebooks and we provide a `.yml` file with required dependencies.

1. **Install Anaconda or Miniconda**:
   If you haven't already, download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Virtual Environment**: Open your terminal (or Anaconda Prompt on Windows) and navigate to the directory containing the OrchAMP.yml file. Then, run the following command to create a virtual environment:
   ```bash
   conda env create -f OrchAMP.yml
   ```
   
3. **Activate the Virtual Environment**:
   Once the environment is created, activate it using the following command:
   ```bash
   conda activate OrchAMP
   ```
   
4. **Run the Jupyter Notebooks**:
   With the environment activated, you can now run the Jupyter notebooks. Start Jupyter Notebook by executing:
   ```bash
   jupyter notebook
   ```
   
This will open the Jupyter Notebook interface in your web browser. From there, you can navigate to and run the .ipynb notebooks in this repository.

## Running the simulation

The repository contains three subfolders `Python_Scripts`, `Slurm_Scripts` and `Jupyter_Notebooks`. 

1. The `Python_Scripts` folder contain the main files that are used to run the simulations. In particular, `effect_of_data_integration.py` is used for generating Table 1 in the manuscript (comparison of data integration methods) and `pred_set.py` is used for generating Table 2 of the manuscript (empirical coverage of the prediction sets).

2. The `Slurm_Scripts` folder contain the scripts used to run the python scripts in the ASC UNITY cluster of Ohio State University. You can open the terminal. Navigate to the repository where the slurm scripts live. You can run `effect_of_data_integration.py` by executing the script `effect_of_data_integration.sh` as follows:
   ```bash
   sbatch effect_of_data_integration.sh
   ```
   
   You can run `pred_set.py` by executing the script `pred_set_experiment.sh` as follows:
   ```bash
   sbatch pred_set_experiment.sh
   ```
   
3. The results from the cluster can be unified by running the Jupyter notebooks `combine_effect_of_data_integration.ipynb` (Table 1) and `combine_calibration_results.ipynb` (Table 2).

## TEA-seq data analysis

### Data Source and Pre-processing Guide

The raw TEA-seq dataset from [Swanson et al. (2021)](https://elifesciences.org/articles/63632), used to generate Figures 1 and 2 of our paper, is available [here]([https://www.dropbox.com/scl/fo/yu1vydyjhab0yxs9kyhoo/AOQwV-4cDz9GtjTRHmNawNg?rlkey=j7bbsfiwihwrzqzkqh4hj3e87&st=kb766idz&dl=0](https://www.dropbox.com/home/PhD%20Projects/Current_Projects_with_Seniors/AMP%2BMultimodal_Data/Data_codes_cmp_norm/Tea_seq_detailed_ablation_3pct/Preprocessing)).

The dataset includes:
- `adt_counts.csv`: Contains raw ADT counts (Protein modality).
- `feature_matrix.h5`: Contains raw RNA and ATAC counts.

These datasets need to be pre-processed using standard Seurat techniques, as described [here](https://www.sciencedirect.com/science/article/pii/S0092867421005833). The pre-processing steps involve:

- **Filtering**: Removing cells and genes with extremely low expression or highly expressed mitochondrial genes.
- **Normalization and Transformation**: Normalizing RNA and ATAC counts and applying a $\log(1 + x)$ transformation.
- **Selection**: Choosing highly variable genes, ATAC reads, and proteins for analysis.

For a detailed explanation of the pre-processing steps, please refer to our paper.
To pre-process the data, run the provided R script `Pre-processing_tea_seq_data.R` located in the `Code` folder. After running the script you will get the following pre-processed datasets.

- `cleaned_adt_tea_seq.csv`: Provides the pre-processed ADT counts (Protein modality).
- `cleaned_atac_reads_tea_seq.h5ad`: Provides the pre-processed ATAC counts.
- `cleaned_rna_reads_tea_seq.h5ad`: Provides the pre-processed RNA counts.
- `cleaned_cell_labels_meta_tea_seq.csv`: Provides the cell-type of the cells used in the analysis.

### Multimodal Cell Atlas Building Using OrchAMP (Algorithm 1)

Using the cleaned datasets, you can build a multimodal cell atlas as described in Section 4.1 of our paper by running the Jupyter notebook `tea_seq_integration_analysis_gmm.ipynb` located in the `tea_seq` folder. The OrchAMP atlas is compared with the atlas produced by the WNN-based integration technique of [Hao et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0092867421005833).

The PCA embeddings required for the [WNN-based integration technique](https://www.sciencedirect.com/science/article/pii/S0092867421005833), is also compute by `tea_seq_integration_analysis_gmm.ipynb`. We also benchmark against [MOFA+](https://pubmed.ncbi.nlm.nih.gov/32393329/). The `MOFA+` embeddings can be obtained by running `tea_seq_mofa_integration.R`.

To generate the UMAP plot shown in Figure 1 of our paper, execute the R script `plot_for_tea_seq.R` found in the `tea_seq` folder.

### Querying Multimodal Atlas with New Cells (Algorithm 2)

Using Algorithm 2 from our paper, you can build point predictions of the embeddings for new query cells with partially observed modalities relative to the constructed multimodal atlas. Additionally, you can create a prediction set with the desired confidence level for such embeddings using the same algorithm.

We illustrate this functionality by constructing a 95% prediction set for three test cells of different categories:
- `pre-B cell` 
- `Double Negative T cell` 
- `CD8 Effector cell` 

The prediction sets are visualized against the UMAP of the reference atlas through 500 points randomly sampled from them. For details of the visualization, please check our paper.

To reproduce our results, you can run the Jupyter notebook `tea_seq_prediction.ipynb`. To reproduce Figure 2 (`Double Negative T cell`) from our paper, execute the R script `prediction_plot_dnt.R` located in the `tea_seq` folder. Similar plots for `pre-B cell` can be obtained by running `prediction_plot_pbc.R` and `CD8 Effector cell` by running `prediction_plot_cd8e.R`.

