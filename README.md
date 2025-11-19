# Moran_Imaging

[![License](https://img.shields.io/pypi/l/Moran_Imaging.svg?color=green)](https://github.com/vandeplaslab/Moran_Imaging/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/Moran_Imaging.svg?color=green)](https://pypi.org/project/Moran_Imaging)
[![Python Version](https://img.shields.io/pypi/pyversions/Moran_Imaging.svg?color=green)](https://python.org)
[![CI](https://github.com/vandeplaslab/Moran_Imaging/actions/workflows/ci.yml/badge.svg)](https://github.com/vandeplaslab/Moran_Imaging/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vandeplaslab/Moran_Imaging/branch/master/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/Moran_Imaging)

The Moran Imaging Python package was developed for the spatio-molecular analysis of multiplexed molecular imaging data, such as imaging mass spectrometry (IMS) data. It provides the statistical tools to identify, visualize, and quantify spatial patterns in molecular imaging data, such as Moran's I and the Moran quadrant map. It also provides user-friendly implementations of a novel tissue domain segmentation workflow, called Moran-Felsenszwalb segmentation, and of a novel colocalization-based image clustering workflow, called Moran-HOG clustering.

### Installation

You can install Moran Imaging, with its mandatory dependencies, from PyPI. Our package supports Python version 3.9 and above. It has been tested on Windows and Linux. The necessary dependencies are listed in pyproject.toml. 

```bash
pip install Moran_Imaging[demo]
```

### Data download

Download the following three datasets from Zenodo: Demo_rat_brain_data.pickle, which is a subset of 100 ion images of IMS dataset no1, Zebra_fish_8_clusters_dataset.pickle, which is a subset of 174 ion images of IMS dataset no2, and Zebra_fish_UMAP_dataset.pickle, which is a UMAP embedding of IMS dataset no2. Dataset no1 is obtained from the coronal section of a Parkinson’s disease rat model. Dataset no2 is obtained from a whole-body adult male zebrafish section. 
Zenodo link: https://zenodo.org/records/17399931

### Tutorials

The paper corresponding to the Moran Imaging Python package is "Spatial Dependence and Heterogeneity in Molecular Imaging: Moran Quadrant Maps Enable Advanced Spatial-Statistical Analysis" by Léonore Tideman, Felipe Moser, Lukasz Migas, Jacquelyn Spathies, Katerina  Djambazova, Cody Marshall, Matthew Schrag, Eric Skaar, Jeffrey Spraggins, Raf Van de Plas (October 2025). Download the tutorial notebooks from GitHub, open the JupyterLab interactive development environment, and run the following notebooks to reproduce our results. 

Run the following notebook to reproduce Figures 1 and 2 of the main manuscipt, and Figures 9 and 10 of the supplementary material. We demonstrate how to quantify spatial dependence and spatial heterogeneity, and how to compute the Moran quadrant map of an image. Please note that automatic parallelization with the `numba` @jit decorator is only available on 64-bit platforms.

        jupyter lab Exploratory_spatial_data_analysis.ipynb 

Run the following notebook to reproduce Figure 4 of the main manuscript, and Figures 11 and 12 of the supplementary material. We propose a computationally efficient parallelized implementation of the Moran-Felsenszwalb segmentation workflow. 

        jupyter lab Moran_Felsenszwalb_segmentation.ipynb
        jupyter lab Fast_Moran_Felsenszwalb_segmentation.ipynb

Run the following notebooks to reproduce the results of Table 1 of the main manuscript. Run the notebook about Moran-HOG clustering to reproduce Figures 14 to 21 of the supplementary material. We recommend running the DeepION and NRDC deep clustering workflows on a GPU. 
 
        jupyter lab Moran_HOG_clustering.ipynb
        jupyter lab DeepION_clustering.ipynb
        jupyter lab NRDC_clustering.ipynb


### Citation

If you reuse our code, please cite our work: "Spatial Dependence and Heterogeneity in Molecular Imaging: Moran Quadrant Maps Enable Advanced Spatial-Statistical Analysis" by Tideman et al. (October 2025).
BioRxiv link: https://www.biorxiv.org/content/10.1101/2025.10.27.684518v1. 
