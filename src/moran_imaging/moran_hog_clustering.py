"""Moran-HOG-Spatial Clustering (MHSClustering) is a class that performs clustering on a dataset of ion images."""

from __future__ import annotations

import numpy as np
from skimage.feature import hog
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

# Import necessary code
from moran_imaging.local_global_spatial_autocorrelation import MoranLocal
from moran_imaging.spatial_weights_matrix import define_spatial_weights_matrix


class MHSClustering:
    """Moran-HOG-Spatial Clustering (MHSClustering) is a class that performs clustering on a dataset of ion images
    using a combination of Moran's I, Histogram of Oriented Gradients (HOG), and spatial clustering.
    """

    # Attributes
    labels: np.ndarray | None = None

    def __init__(
        self,
        dataset,
        acquisition_mask,
        image_shape: tuple[int, int],
        neighborhood_size: int = 5,
        hog_orientations: int = 8,
        hog_pixels_per_cell: tuple[int, int] = (16, 16),
        hog_cells_per_block: tuple[int, int] = (1, 1),
        hog_visualize: bool = False,
        random_seed: int = 0,
    ):
        self.random_seed = random_seed
        self.image_shape = image_shape
        self.orientations = hog_orientations
        self.pixels_per_cell = hog_pixels_per_cell
        self.cells_per_block = hog_cells_per_block

        # Define the spatial weights matrix
        background_mask = np.invert(acquisition_mask)
        weights_matrix = define_spatial_weights_matrix(
            image_shape, "Queen", neighborhood_size, background_mask, with_lower_order=True
        )
        weights_matrix.transform = "r"

        # Compute the Moran quadrant maps
        moran_quadrants = np.zeros((image_shape[0] * image_shape[1], dataset.shape[1])).astype(np.float32)
        for mz_index in range(dataset.shape[1]):
            ion_image_with_background = self.reshape_image(dataset[:, mz_index], background_mask)
            local_moran_object = MoranLocal(
                ion_image_with_background.astype(np.float32),
                weights_matrix,
                background_mask,
                num_permute=999,
                num_jobs=-1,
            )
            moran_quadrants_ion_image = local_moran_object.quadrant.astype(np.float32)
            moran_quadrants[:, mz_index] = moran_quadrants_ion_image
        self.Moran_quadrants = moran_quadrants

        # Compute the Histogram of Oriented Gradients (HOG)
        if hog_visualize is True:
            self.Moran_quadrants_HOG_features, self.Moran_quadrants_HOG_images = self.extract_hog_features(
                moran_quadrants
            )
        else:
            self.Moran_quadrants_HOG_features, _ = self.extract_hog_features(moran_quadrants)

    def reshape_image(self, data: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
        """Reshape a 1D array of data into a 2D image using a background mask."""
        pixel_grid = np.zeros((self.image_shape[0] * self.image_shape[1],))
        pixel_grid[np.invert(background_mask)] = data
        image = np.reshape(pixel_grid, self.image_shape)
        return image

    def extract_hog_features(self, dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract Histogram of Oriented Gradients (HOG) features from a 2D array.

        Each row represents a flattened image.
        """
        # Transpose so that each row is a flattened image
        dataset = dataset.transpose()

        total_hog_images = []
        total_hog_features = []

        # Compute a Histogram of Oriented Gradients (HOG)
        for flattened_image in tqdm(dataset, desc="Extracting HOG Features"):
            # Reshape the flattened image back into its 2D form
            ion_image = flattened_image.reshape(self.image_shape)

            # Compute HOG features and the HOG image of the ion image
            hog_features, hog_image = hog(
                ion_image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=True,
                block_norm="L2-Hys",
            )
            total_hog_images.append(hog_image.flatten())
            total_hog_features.append(hog_features)

        # Return all HOG features as a 2D array where each row is a feature vector for an image
        return np.array(total_hog_features), np.array(total_hog_images)

    def clustering_k_means(self, num_clusters: int) -> np.ndarray:
        """Perform clustering using k-means on the HOG features."""
        # Mean-centering and unit-variance scaling
        scaler_model = StandardScaler(with_mean=True, with_std=True)
        hog_features = scaler_model.fit_transform(self.Moran_quadrants_HOG_features)

        # k-means clustering
        kmeans_model = KMeans(n_clusters=num_clusters, n_init=50, random_state=self.random_seed)
        kmeans_model.fit(hog_features)
        self.labels = kmeans_model.labels_
        return self.labels

    def clustering_hdbscan(self, min_cluster_size: int, max_cluster_size: int) -> np.ndarray:
        """Perform clustering using HDBSCAN on the HOG features."""
        # Non-linear dimensionality reduction with UMAP
        umap_model = UMAP(n_neighbors=min_cluster_size, n_components=5, metric="euclidean", init="pca")
        hog_features = umap_model.fit_transform(self.Moran_quadrants_HOG_features)

        # HDBSCAN clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, metric="euclidean", n_jobs=-1
        )
        hdbscan_model.fit(hog_features)
        self.labels = hdbscan_model.labels_
        return self.labels


# Kept for backwards compatibility
Moran_HOG_SKD_clustering = MHSClustering
