"""Plotting functions."""

from __future__ import annotations

import typing as ty

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from moran_imaging.local_global_spatial_autocorrelation import lag_spatial

if ty.TYPE_CHECKING:
    from moran_imaging.local_global_spatial_autocorrelation import MoranLocal
    from moran_imaging.spatial_weights_matrix import SpatialWeightsMatrix


def position_discrete_colorbar_ticks(min_tick: float, num_ticks: int) -> np.ndarray:
    """Determine tick locations for a discrete colorbar given the total number of ticks and the minimum tick value.
    We assume that the total number of ticks is also equal to the maximum tick value.
    """
    step = 1 / (2 * num_ticks)
    tick_location = [step]
    for _index in range(0, num_ticks - 1):
        tick_location.append(tick_location[-1] + 2 * step)
    tick_location = [min_tick + (num_ticks - 1) * x for x in tick_location]
    return np.round(tick_location, 5)


def moran_hot_cold_spots(moran_local: MoranLocal, p: float) -> np.ndarray:
    """Assign each pixel to a quadrant (labelled from one to four) of the Moran scatterplot."""
    if p is not None:
        significance = moran_local.p_sim < p
        high_high = 1 * np.logical_and(significance, moran_local.quadrant == 1)
        low_high = 2 * np.logical_and(significance, moran_local.quadrant == 2)
        low_low = 3 * np.logical_and(significance, moran_local.quadrant == 3)
        high_low = 4 * np.logical_and(significance, moran_local.quadrant == 4)
    else:
        high_high = 1 * (moran_local.quadrant == 1)
        low_high = 2 * (moran_local.quadrant == 2)
        low_low = 3 * (moran_local.quadrant == 3)
        high_low = 4 * (moran_local.quadrant == 4)
    return high_high + low_low + low_high + high_low


# noinspection PyPep8Naming
def plot_moran_local_scatterplot(
    moran_local: MoranLocal,
    W: SpatialWeightsMatrix,
    background_mask: np.ndarray,
    ion_image: np.ndarray,
    mz: float,
    p: float | None = None,
    colormap: str | None = None,
    axes: plt.Axes | None = None,
    with_colorbar: bool = True,
):
    """
    Moran Scatterplot with option of coloring of Local Moran Statistics.

    Inputs
    ----------
    moran_local     : local Moran's I object
    W               : spatial weights matrix object
    background_mask : array
                      Boolean numpy array of missing values of size (one boolean variable per pixel)
    ion_image       : array
                      Attribute spatial distribution (e.g. ion image of one m/z bin).
    mz              : float
                      Mass-to-charge value corresponding to the ion image
    p               : pseudo p-value for significance thresholding.
                      If p=None, there is no significance testing and all points are assigned to a quadrant.
                      Otherwise, points with non-significant spatial autocorrelation are displayed in grey.
    colormap        : matplotlib colormap
                      Colormap with four colors for the four scatterplot quadrants.
    axes            : matplotlib Axes
                      Axes on which to plot the scatterplot.
    with_colorbar   : boolean
                      True to display colorbar; False otherwise.

    Outputs
    -------
    slope            : slope of linear regression line (equivalent to Moran's I)
    intercept        : intercept of linear regression line (equivalent to mean spatial lag)
    R2               : coefficient of determination
    y_mean           : mean attribute (excluding off-tissue pixels if applicable)
    mean_spatial_lag : mean spatial lag
    """
    # Check for off-tissue pixels (background mask)
    remove_background = len(background_mask) != 0

    # Flatten ion image and perform mean-centering
    # Exclude background pixels from mean calculation if applicable
    if remove_background is False:
        y = np.asarray(ion_image).flatten()
        y_mean = y.mean()
    else:
        ion_image_flat = np.asarray(ion_image).flatten()
        y_mean = np.mean(ion_image_flat[np.invert(background_mask)])
        y = ion_image_flat - y_mean
        y[background_mask] = 0

    # Define the colormap of the scatterplot quadrants
    spots = moran_hot_cold_spots(moran_local, p)
    if colormap is None:
        colormap = ListedColormap(["#fb485e", "#91dff6", "#5353ea", "#ffa6d9"])
    if (p is None) or (0 not in spots):
        hmap = colormap
        tick_labels = [
            "High attribute and high spatial lag",
            "Low attribute and high spatial lag",
            "Low attribute and low spatial lag",
            "High attribute and low spatial lag",
        ]
        tick_loc = position_discrete_colorbar_ticks(1, 4)
    else:
        hmap = ["#bababa", *colormap]
        tick_labels = [
            "Non-significant spatial autocorrelation",
            "High attribute and high spatial lag",
            "Low attribute and high spatial lag",
            "Low attribute and low spatial lag",
            "High attribute and low spatial lag",
        ]
        tick_loc = position_discrete_colorbar_ticks(0, 5)

    # Parameters for plotting the scatter points and the linear regression line
    scatter_kwds = {}
    scatter_kwds.setdefault("alpha", 0.6)
    scatter_kwds.setdefault("s", 40)
    fitline_kwds = {}
    fitline_kwds.setdefault("alpha", 0.9)

    # Create Matplotlib figure and axes
    if axes is None:
        fig, axes = plt.subplots(figsize=(7, 7), dpi=100)

    # Set figure labels and title
    axes.set_xlabel("Intensity", fontsize=12)
    axes.set_ylabel("Spatial Lag", fontsize=12)
    axes.set_title("Moran local scatterplot of " + str(mz))

    # Perform a bivariate regression (ordinary least squares linear regression) of the attribute on its spatial lag.
    # The slope is equal to Moran's I. The intercept is equal to the mean spatial lag.
    lag = lag_spatial(W, y)
    mean_spatial_lag = lag.mean()
    OLS_results = np.polyfit(y, lag, 1, full=True)
    slope, intercept = OLS_results[0]

    # Compute the coefficient of determination R^2
    RSS = OLS_results[1][0]  # residual sum of squares
    square_diff = (lag - mean_spatial_lag) ** 2
    TSS = square_diff.sum()  # total sum of squares
    R2 = 1 - RSS / TSS

    # Plot a horizontal line at the mean of the spatially lagged attribute values
    # Plot a vertical line at the mean of the attribute values
    axes.hlines(mean_spatial_lag, y_mean + 1.1 * y.min(), y_mean + 1.1 * y.max(), alpha=0.5, linestyle="--", color="k")
    axes.vlines(
        y_mean,
        mean_spatial_lag + 1.1 * lag.min(),
        mean_spatial_lag + 1.1 * lag.max(),
        alpha=0.5,
        linestyle="--",
        color="k",
    )

    # Plot scatter plot of the attribute (in deviations from the mean) versus the spatial lag
    # Plot linear fit, whose slope is the global Moran's I statistic
    fitline_kwds.setdefault("color", "k")
    scatter_kwds.setdefault("cmap", hmap)

    if remove_background is False:
        scatter_kwds.setdefault("c", spots)
        image_scatter = axes.scatter(y, lag, **scatter_kwds)
        axes.plot(y_mean + y, mean_spatial_lag + slope * y, linewidth=2, **fitline_kwds)
    else:
        y_no_background = y[np.invert(background_mask)]
        lag_no_background = lag[np.invert(background_mask)]
        spots_no_background = spots[np.invert(background_mask)]
        scatter_kwds.setdefault("c", spots_no_background)
        image_scatter = axes.scatter(y_mean + y_no_background, mean_spatial_lag + lag_no_background, **scatter_kwds)
        axes.plot(y_mean + y_no_background, mean_spatial_lag + slope * y_no_background, linewidth=2, **fitline_kwds)

    # Colorbar
    if with_colorbar is True:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(image_scatter, cax, ticks=tick_loc)
        cbar.ax.set_yticklabels(tick_labels, fontsize=10)
        cbar.solids.set(alpha=1)

    return slope, intercept, R2, y_mean, mean_spatial_lag
