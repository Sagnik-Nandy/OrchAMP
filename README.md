# Multimodal Atlas Building and Query prediction using Orchestrated Approximate Message Passing

This GitHub repository contains codes for implementing the techniques for building multimodal cell atlas and querying new subjects that are only partially observed via the algorithms proposed in [Multimodal data integration and cross-modal querying via orchestrated approximate message passing](https://arxiv.org/abs/2407.19030) by Nandy and Ma (2024). 

## Installation Guide

To run the Jupyter notebooks in this repository, you need to set up a Python environment using the `OrchAMP.yml` file. Follow these steps to create and activate the virtual environment:

1. **Install Anaconda or Miniconda**:
   If you haven't already, download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Virtual Environment**:
   Open your terminal (or Anaconda Prompt on Windows) and navigate to the directory containing the `OrchAMP.yml` file. Then, run the following command to create a virtual environment:
   ```bash
   conda env create -f OrchAMP.yml

3. **Activate the Virtual Environment**:
   Once the environment is created, activate it using the following command:
   ```bash
   conda activate OrchAMP

4. **Run the Jupyter Notebooks**:
   With the environment activated, you can now run the Jupyter notebooks. Start Jupyter Notebook by executing:
   ```bash
   jupyter notebook
   
This will open the Jupyter Notebook interface in your web browser. From there, you can navigate to and run the .ipynb notebooks in this repository.

## Data Source and Pre-processing Guide

The raw TEA-seq dataset from [Swanson et al. (2021)](https://elifesciences.org/articles/63632), used to generate Figures 1 and 2 of our paper, is available [here](https://www.dropbox.com/scl/fo/yu1vydyjhab0yxs9kyhoo/AOQwV-4cDz9GtjTRHmNawNg?rlkey=j7bbsfiwihwrzqzkqh4hj3e87&st=kb766idz&dl=0).

The dataset includes:
- `adt_counts.csv`: Contains raw ADT counts (Protein modality).
- `feature_matrix.h5`: Contains raw RNA and ATAC counts.

These datasets need to be pre-processed using standard Seurat techniques, as described [here](https://www.sciencedirect.com/science/article/pii/S0092867421005833). The pre-processing steps involve:

- **Filtering**: Removing cells and genes with extremely low expression or highly expressed mitochondrial genes.
- **Normalization and Transformation**: Normalizing RNA and ATAC counts and applying a $\log(1 + x)$ transformation.
- **Selection**: Choosing highly variable genes, ATAC reads, and proteins for analysis.

For a detailed explanation of the pre-processing steps, please refer to our paper.
To pre-process the data, run the provided R script `Pre-processing_tea_seq_data.R` located in the `Codes` folder. After running the script you will get the following pre-processed datasets.

- `cleaned_adt_tea_seq.csv`: Provides the pre-processed ADT counts (Protein modality).
- `cleaned_atac_reads_tea_seq.h5ad`: Provides the pre-processed ATAC counts.
- `cleaned_rna_reads_tea_seq.h5ad`: Provides the pre-processed RNA counts.
- `cleaned_cell_labels_meta_tea_seq.csv`: Provides the cell-type of the cells used in the analysis.

## Multimodal Cell Atlas Building Using OrchAMP (Algorithm 1)

Using the cleaned datasets, you can build a multimodal cell atlas as described in Section 3.2 of our paper by running the Jupyter notebook `tea_seq_atlas_building.ipynb` located in the `Codes` folder. The OrchAMP atlas is compared with the atlas produced by the WNN-based integration technique of [Hao et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0092867421005833).

To compute the PCA embeddings required for the [WNN-based integration technique](https://www.sciencedirect.com/science/article/pii/S0092867421005833), run the notebook `tea_seq_atlas_building_wnn.ipynb`.

To generate the UMAP plot shown in Figure 1 of our paper, execute the R script `tea_seq_atlas_plotting.R` found in the `Codes` folder.

## Querying Multimodal Atlas with New Cells (Algorithm 2)

Using Algorithm 2 from our paper, you can build point predictions of the embeddings for new query cells with partially observed modalities relative to the constructed multimodal atlas. Additionally, you can create a prediction set with the desired confidence level for such embeddings using the same algorithm.

We illustrate this functionality by constructing a 95% prediction set for three test cells of different categories:
- `pre-B cell` (using the `Protein` modality)
- `Double Negative T cell` (using the `RNA` modality)
- `CD8 Effector cell` (using the `ATAC` modality)

The prediction sets are visualized against the UMAP of the reference atlas through 500 points randomly sampled from them. 

To reproduce our results, you can run the Jupyter notebook `tea_seq_prediction.ipynb`. To reproduce Figure 2 from our paper, execute the R script `tea_seq_prediction_plotting.R` located in the `Codes` folder. 

