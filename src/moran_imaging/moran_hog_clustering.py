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

    def __init__(
        self,
        dataset,
        acquisition_mask,
        image_shape,
        neighborhood_size=5,
        HOG_orientations=8,
        HOG_pixels_per_cell=(16, 16),
        HOG_cells_per_block=(1, 1),
        HOG_visualize=False,
        random_seed=0,
    ):
        self.random_seed = random_seed
        self.image_shape = image_shape
        self.orientations = HOG_orientations
        self.pixels_per_cell = HOG_pixels_per_cell
        self.cells_per_block = HOG_cells_per_block

        # Define the spatial weights matrix
        background_mask = np.invert(acquisition_mask)
        weights_matrix = define_spatial_weights_matrix(
            image_shape, "Queen", neighborhood_size, background_mask, with_lower_order=True
        )
        weights_matrix.transform = "r"

        # Compute the Moran quadrant maps
        Moran_quadrants = np.zeros((image_shape[0] * image_shape[1], dataset.shape[1])).astype(np.float32)
        for mz_index in range(dataset.shape[1]):
            ion_image_with_background = self.reshape_image(dataset[:, mz_index], background_mask)
            local_Moran_object = MoranLocal(
                ion_image_with_background.astype(np.float32),
                weights_matrix,
                background_mask,
                num_permute=999,
                num_jobs=-1,
            )
            Moran_quadrants_ion_image = local_Moran_object.quadrant.astype(np.float32)
            Moran_quadrants[:, mz_index] = Moran_quadrants_ion_image
        self.Moran_quadrants = Moran_quadrants

        # Compute the Histogram of Oriented Gradients (HOG)
        if HOG_visualize is True:
            self.Moran_quadrants_HOG_features, self.Moran_quadrants_HOG_images = self.extract_hog_features(
                Moran_quadrants
            )
        else:
            self.Moran_quadrants_HOG_features, _ = self.extract_hog_features(Moran_quadrants)

    def reshape_image(self, data, background_mask):
        pixel_grid = np.zeros((self.image_shape[0] * self.image_shape[1],))
        pixel_grid[np.invert(background_mask)] = data
        image = np.reshape(pixel_grid, self.image_shape)
        return image

    def extract_hog_features(self, dataset):
        """Extract Histogram of Oriented Gradients (HOG) features from a 2D array where each row represents a flattened image."""
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

    def clustering_k_means(self, num_clusters):
        # Mean-centering and unit-variance scaling
        scaler_model = StandardScaler(with_mean=True, with_std=True)
        hog_features = scaler_model.fit_transform(self.Moran_quadrants_HOG_features)

        # k-means clustering
        kmeans_model = KMeans(n_clusters=num_clusters, n_init=50, random_state=self.random_seed)
        kmeans_model.fit(hog_features)
        self.labels = kmeans_model.labels_

        return self.labels

    def clustering_hdbscan(self, min_cluster_size, max_cluster_size):
        # Non-linear dimensionality reduction with UMAP
        UMAP_model = UMAP(n_neighbors=min_cluster_size, n_components=5, metric="euclidean", init="pca")
        hog_features = UMAP_model.fit_transform(self.Moran_quadrants_HOG_features)

        # HDBSCAN clustering
        HDBSCAN_model = HDBSCAN(
            min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, metric="euclidean", n_jobs=-1
        )
        HDBSCAN_model.fit(hog_features)
        self.labels = HDBSCAN_model.labels_

        return self.labels


# Kept for backwards compatibility
Moran_HOG_SKD_clustering = MHSClustering
