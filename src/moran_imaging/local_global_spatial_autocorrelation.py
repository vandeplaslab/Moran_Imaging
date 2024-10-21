# Local and global spatial autocorrelation

import numpy as np
from .centralized_conditional_randomization_engine import *
from .spatial_weights_matrix import *


def lag_spatial(W, y):
    """
    Spatial lag operator. If W is row-standardized, compute the weighted mean of each observation's neighbors. If not, compute the weighted sum of each observation's neighbors.

    Inputs
    ----------
    W  : spatial weights matrix object
    y  : array

    Output
    -------
    Wy : array
    """
    Wy = W.sparse * y

    return Wy


class Moran_Local:
    """Local Moran's I statistics.

    Parameters
    ----------
    ion_image         : array (required data type: float64)
                        Attribute spatial distribution (e.g. ion image of one m/z bin).
    W                 : spatial weights matrix object
    background_mask   : array
                        Boolean numpy array of missing values of size (one boolean per pixel)
    num_permute       : integer
                        Number of random permutations for calculating the pseudo p-values.
    significance_test : boolean
                        True to do test Moran's I statistics' significance by local randomization; False otherwise.
                        Note that local significance testing is computationally costly.
    num_jobs          : integer
                        Number of cores to be used in the conditional randomization. If -1, all available cores are used.

    Attributes
    ----------
    Moran_I_local    : array
                       local Moran's I statistics.
    quadrant         : array
                       values indicate quandrant location - 1 (high-high), 2 (low-high), 3 (low-low), 4 (high-low).
    sim              : array
                       local Moran's I statistics obtained for randomly permuted images.
    p_sim            : array
                       One-sided local pseudo p-values based on pixel permutations.
    EI_sim           : array
                       Average values of local Moran's I statistics from permutations.
    seI_sim          : array
                       Standard deviations of local Moran's I statistics under permutations.
    VI_sim           : array
                       Variance of local Moran's I statistics from permutations.
    z_sim            : arrray
                       Standardized local Moran's I statistics based on permutations.
    """

    def __init__(self, ion_image, W, background_mask, num_permute=999, significance_test=False, num_jobs=-1):

        # Flatten ion image and perform mean-centering (excluding background pixels)
        ion_image_flat = np.asarray(ion_image).flatten()
        if len(background_mask) == 0:
            z = ion_image_flat - ion_image_flat.mean()
        else:
            mean_no_background = np.mean(ion_image_flat[np.invert(background_mask)])
            z = ion_image_flat - mean_no_background
            z[background_mask] = 0

        # Compute local Moran's I statistics
        self.Moran_I_local = self._compute_local_Moran_I(W, z)

        # Assign each pixel to a quadrant of the Moran scatterplot
        self.quadrant = self._determine_scatterplot_quadrant(W, z, [1, 2, 3, 4])

        if significance_test == True:
            # Simulate spatial randomness by computing the local Moran's I statistics corresponding to random permutations of the image
            # Numba-accelerated parallelized conditional randomization
            _, rlisas = crand(
                z,
                W,
                self.Moran_I_local,
                num_permute,
                keep=True,
                n_jobs=num_jobs,
                stat_func=moran_local_crand,
                seed=None,
            )
            self.sim = np.transpose(rlisas)

            # Compute pseudo p-values for assessing the significance of local spatial autocorrelation statistics
            above = self.sim >= self.Moran_I_local
            larger = above.sum(0)
            low_extreme = (num_permute - larger) < larger
            larger[low_extreme] = num_permute - larger[low_extreme]
            self.p_sim = (larger + 1.0) / (num_permute + 1.0)

            # Summarize the distribution of local Moran's I statistics obtained from random permutations
            self.EI_sim = self.sim.mean(axis=0)
            self.seI_sim = self.sim.std(axis=0)
            self.VI_sim = self.seI_sim * self.seI_sim
            self.z_sim = (self.Moran_I_local - self.EI_sim) / (self.seI_sim + 1e-12)

    def _compute_local_Moran_I(self, W, z):
        z_lag = lag_spatial(W, z)
        numerator = z * z_lag
        denominator = (z * z).sum()
        Is = numerator / denominator
        return Is

    def _determine_scatterplot_quadrant(self, W, z, quads):
        z_lag = lag_spatial(W, z)
        z_pos = z > 0
        lag_pos = z_lag > 0
        pos_pos = z_pos * lag_pos
        neg_pos = (1 - z_pos) * lag_pos
        neg_neg = (1 - z_pos) * (1 - lag_pos)
        pos_neg = z_pos * (1 - lag_pos)
        q = quads[0] * pos_pos + quads[1] * neg_pos + quads[2] * neg_neg + quads[3] * pos_neg
        return q


class Moran_Global:
    """Global Moran's I statistics.

    Parameters
    ----------
    ion_image         : array (required data type: float64)
                        Attribute spatial distribution (e.g. ion image of one m/z bin).
    W                 : spatial weights matrix object
    background_mask   : array
                        Boolean numpy array of missing values of size (one boolean per pixel)
    significance_test : boolean
                        True to do test Moran's I significance by global randomization; False otherwise.
    num_permute       : integer
                        Number of random permutations for calculating the pseudo p-values

    Attributes
    ----------
    Moran_I_global    : array
                        Global Moran's I statistics.
    sim               : array
                        Global Moran's I statistics obtained for randomly permuted images.
    p_sim             : array
                        One-sided global pseudo p-values based on pixel permutations.
    EI_sim            : array
                        Average values of global Moran's I statistics from permutations.
    seI_sim           : array
                        Standard deviations of global Moran's I statistics under permutations.
    VI_sim            : array
                        Variance of global Moran's I statistics from permutations.
    z_sim             : arrray
                        Standardized global Moran's I statistics based on permutations.
    """

    def __init__(self, ion_image, W, background_mask, num_permute=999):
        # Flatten ion image and perform mean-centering (excluding background pixels from the mean calculation)
        ion_image_flat = np.asarray(ion_image).flatten()
        if len(background_mask) == 0:
            z = ion_image_flat - ion_image_flat.mean()
        else:
            mean_no_background = np.mean(ion_image_flat[np.invert(background_mask)])
            z = ion_image_flat - mean_no_background
            z[background_mask] = 0

        # Compute ratio of the number of pixels (excluding background pixels) to the sum of the weights
        factor = (W.n - len(W.islands)) / W.s0

        # Compute global Moran's I
        self.Moran_I_global = self._compute_global_Moran_I(z, W, factor)

        # Simulate spatial randomness by computing the global Moran's I statistics corresponding to random permutations of the image
        sim = [self._compute_global_Moran_I(np.random.permutation(z), W, factor) for i in range(num_permute)]
        self.sim = np.array(sim)

        # Compute pseudo p-values for assessing the significance of global spatial autocorrelation statistics
        above = self.sim >= self.Moran_I_global
        larger = above.sum()
        if (num_permute - larger) < larger:
            larger = num_permute - larger
        self.p_sim = (larger + 1.0) / (num_permute + 1.0)

        # Summarize the distribution of global Moran's I statistics obtained from random permutations
        self.EI_sim = self.sim.sum() / num_permute
        self.seI_sim = np.array(self.sim).std()
        self.VI_sim = self.seI_sim**2
        self.z_sim = (self.Moran_I_global - self.EI_sim) / (self.seI_sim + 1e-12)

    def _compute_global_Moran_I(self, z, W, factor):
        z_lag = lag_spatial(W, z)
        inum = (z * z_lag).sum()
        z2ss = (z * z).sum()
        I = factor * (inum / z2ss)
        return I


class Geary_Global:
    """Global Geary's c statistics.

    Parameters
    ----------
    y              : array
                     Attribute spatial distribution (e.g. ion image of one m/z bin)
    W              : spatial weights matrix object
    num_permute    : integer
                     Number of random permutations for calculating the pseudo p-values

    Attributes
    ----------
    Geary_c_global : array
                     Global Geary's c statistics.
    sim            : array
                     Global Geary's c statistics obtained for randomly permuted images.
    p_sim          : array
                     One-sided global pseudo p-values based on pixel permutations.
    EC_sim         : array
                     Average values of global Geary's c statistics from permutations.
    seC_sim        : array
                     Standard deviations of global Geary's c statistics under permutations.
    VC_sim         : array
                     Variance of global Geary's c statistics from permutations.
    z_sim          : arrray
                     Standardized global Geary's c statistics based on permutations.
    """

    def __init__(self, y, W, num_permute=999):
        # Flatten and center the image
        y = np.asarray(y).flatten()

        # Compute global Geary's c
        self.Geary_c_global = self._compute_global_Geary_c(y, W)

        # Simulate spatial randomness by computing the global Moran's I statistics corresponding to random permutations of the image
        sim = [self._compute_global_Geary_c(np.random.permutation(y), W) for i in range(num_permute)]
        self.sim = np.array(sim)

        # Compute pseudo p-values for assessing the significance of global spatial autocorrelation statistics
        above = self.sim >= self.Geary_c_global
        larger = above.sum()
        if (num_permute - larger) < larger:
            larger = num_permute - larger
        self.p_sim = (larger + 1.0) / (num_permute + 1.0)

        # Summarize the distribution of global Geary's c statistics obtained from random permutations
        self.EC_sim = self.sim.sum() / num_permute
        self.seC_sim = np.array(self.sim).std()
        self.VC_sim = self.seC_sim**2
        self.z_sim = (self.Geary_c_global - self.EC_sim) / (self.seC_sim + 1e-12)

    def _compute_global_Geary_c(self, y, W):
        # Numerator
        focal_ix, neighbor_ix = W.sparse.nonzero()
        numerator = (W.sparse.data * ((y[focal_ix] - y[neighbor_ix]) ** 2)).sum()
        numerator = (W.n - len(W.islands) - 1) * numerator

        # Denominator
        z = y - y.mean()
        denominator = sum(z * z) * W.s0 * 2.0

        return numerator / denominator
