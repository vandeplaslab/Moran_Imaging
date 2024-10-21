"""Spatial analysis of molecular imaging data: spatial dependence and heterogeneity statistics, spatial segmentation, spatial clustering."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("Moran_Imaging")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Leonore Tideman"
__email__ = "leonoortideman@gmail.com"
