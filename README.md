# Moran_Imaging

[![License](https://img.shields.io/pypi/l/Moran_Imaging.svg?color=green)](https://github.com/LEMTideman/Moran_Imaging/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/Moran_Imaging.svg?color=green)](https://pypi.org/project/Moran_Imaging)
[![Python Version](https://img.shields.io/pypi/pyversions/Moran_Imaging.svg?color=green)](https://python.org)
[![CI](https://github.com/LEMTideman/Moran_Imaging/actions/workflows/ci.yml/badge.svg)](https://github.com/LEMTideman/Moran_Imaging/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LEMTideman/Moran_Imaging/branch/main/graph/badge.svg)](https://codecov.io/gh/LEMTideman/Moran_Imaging)

Spatial analysis of molecular imaging data: spatial dependence and heterogeneity statistics, spatial segmentation, spatial clustering.

### Installation from PyPI

You can install Moran_Imaging from PyPI:

```bash
pip install Moran_Imaging
```

### Citation

If you reuse our code, please cite our work **provide link to preprint**. 
For academic publications, please use the following Bibtex entry:

**Insert a bibtex-like citation hereunder**

	@article{polanski2019bbknn,
	  title={BBKNN: Fast Batch Alignment of Single Cell Transcriptomes},
	  author={Pola{\'n}ski, Krzysztof and Young, Matthew D and Miao, Zhichao and Meyer, Kerstin B and Teichmann, Sarah A and Park, Jong-Eun},
	  doi={10.1093/bioinformatics/btz625},
	  journal={Bioinformatics},
	  year={2019}
	}


### Figure reproduction

The paper corresponding to the Moran_Imaging Python package is "Imaging Mass Spectrometry: A Spatial Perspective" by Leonore Tideman, Lukasz G. Migas, Katerina V. Djambazova, Jacquelyn Spathies, Jeffrey M. Spraggins, and Raf Van de Plas (2024). Follow our instructions to reproduce our results. 

1. Install the Moran_Imaging package from PyPI.

        pip install Moran_Imaging

3. Download the following three imaging mass spectrometry datasets from Zenodo: Demo_rat_brain_data.pickle, Zebra_fish_8_clusters_dataset.pickle, Zebra_fish_PCA_dataset.pickle. **provide link to zenodo**

5. Download the following three tutorial notebooks: Demo_notebook_exploratory_analysis.ipynb, Demo_notebook_segmentation.ipynb, Demo_notebook_clustering.ipynb.

7. Define two subfolders: one subfolder should be called Data and the other should be called Figures. Save the three datasets in the Data subfolder and leave the Figures subfolder empty.
   
9. Open the JupyterLab interactive development environment, and run the three following Jupyter notebooks.

        jupyter lab Demo_notebook_exploratory_analysis.ipynb 
        jupyter lab Demo_notebook_segmentation.ipynb
        jupyter lab Demo_notebook_clustering.ipynb

Run the Demo_notebook_exploratory_analysis.ipynb notebook to reproduce figures 3, 4, 5, 6, S1 and S2 of our paper. Run the Demo_notebook_segmentation.ipynb notebook to reproduce figures 7, 8, and 9. Run the Demo_notebook_clustering.ipynb notebook to reproduce Figure 12, S3, S4, S5, S6, S7, S8, S9, and S10. Please note that running the deep clustering section of the Demo_notebook_clustering.ipynb notebook may be time-consuming.
