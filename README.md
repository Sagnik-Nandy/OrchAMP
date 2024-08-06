# Codes for implementing atlas building and cross-modal query prediction described in 'Multimodal data integration and cross-modal querying via orchestrated approximate message passing' by Nandy and Ma (2024)

This GitHub repository contains codes for implementing the techniques for building multimodal cell atlas and querying new subjects that are only partially observed via the algorithms proposed in [Multimodal data integration and cross-modal querying via orchestrated approximate message passing](https://arxiv.org/abs/2407.19030) by Nandy and Ma (2024). 

## Installation Guide

To run the Jupyter notebooks in this repository, you need to set up a Python environment using the `OrchAMP.yml` file. Follow these steps to create and activate the virtual environment:

1. **Install Anaconda or Miniconda**:
   If you haven't already, download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Virtual Environment**:
   Open your terminal (or Anaconda Prompt on Windows) and navigate to the directory containing the `OrchAMP.yml` file. Then, run the following command to create a virtual environment:
   ```bash
   conda env create -f OrchAMP.yml
