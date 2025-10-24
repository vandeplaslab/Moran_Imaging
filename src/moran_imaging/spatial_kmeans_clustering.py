# Spatial k-means clustering

# Paper: "Efficient spatial segmentation of large imaging mass spectrometry datasets with spatially aware clustering"
# by Theodore Alexandrov and Jan Hendrik Kobarg. 2001, Bioinformatics, DOI: 10.1093/bioinformatics/btr246.

# Our code is adapted from the open-source MsiFlow Python package, by Philippa Spangenberg and Devon Siemes
# Paper: "msiFlow: Automated Workflows for Reproducible and Scalable Multimodal Mass Spectrometry Imaging and Immunofluorescence
# Microscopy Data Processing and Analysis" by Philippa Spangenberg et al. DOI: https://doi.org/10.1101/2024.08.24.609403
# Original code: https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/b41d98d85a2cf1c247badc65872332a684fdd220/pkg/SA.py

import math
import random
from math import pow

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def choose_distant_objects(col):
    """
    Choose the pivot objects Oa and Ob for which the distance is maximized to create a line on which objects can
    be projected.
    :return: the desired pair of pivot objects with the maximal distance.
    """
    global X

    # choose arbitrarily an object, and declare it to be the second pivot object obj_b
    obj_b = random.randint(0, X.shape[0] - 1)

    # set obj_a = object that is farthest apart from obj_b according to distance function
    # calculate distances from Ob to all other objects
    dists_to_obj_b = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        dists_to_obj_b[i] = distance_between_projections(i, obj_b, col)
    # select object with maxiumum distance as obj_b
    obj_a = np.argmax(dists_to_obj_b)

    # set obj_b = object that is farthest apart from obj_a
    dists_to_obj_a = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        dists_to_obj_a[i] = distance_between_projections(i, obj_a, col)
    # select object with maxiumum distance as obj_b
    obj_b = np.argmax(dists_to_obj_a)

    # report objects Oa and Ob as the desired pair of objects
    return obj_a, obj_b


def distance_between_projections(i, j, col):
    """
    :param i: first object id
    :param j: second object id
    :return: distance between Oi and Oj according to equation 4 of Fastmap
    """
    global data_matrix
    global X

    if col == 0:
        return np.linalg.norm(data_matrix[j] - data_matrix[i])
    else:
        res = math.pow(distance_between_projections(i, j, col - 1), 2) - (math.pow(X[i][col - 1] - X[j][col - 1], 2))
        return math.sqrt(np.abs(res))


def Fastmap(k):
    """
    Find N points in k-d space whose euclidean distances match the distances of a given NxN distance matrix.
    Projection of objects as points in a n-dimensional space on k mutually directions.
    :param k: desired number of dimensions.
    """
    global data_matrix  # original input data
    global X  # output with the ith row being the image of the ith object at the end of the algorithm
    global PA  # stores the ids of the pivot objects - one pair per recursive call
    global col  # stores the column of the X array currently being updated

    if k <= 0:
        return
    else:
        col += 1

    # Choose pivot objects
    obj_a, obj_b = choose_distant_objects(col)

    # Record the ids of the pivot objects
    PA[0, col] = obj_a
    PA[1, col] = obj_b

    # If distance between pivot objects is 0, set X[i, col] = 0 because all inter-object distances are 0
    if distance_between_projections(obj_a, obj_b, col) == 0:
        for i in range(X.shape[0]):
            X[i, col] = 0
    # Project the objects on the line (obj_a, obj_b) for each obj_i according equation 3
    else:
        for i in range(X.shape[0]):
            # projection of first pivot is always 0
            if i == obj_a:
                X[i, col] = 0
            # projection of second pivot is always its distance from the first
            elif i == obj_b:
                X[i, col] = distance_between_projections(obj_a, obj_b, col)
                # print("dist=", distance_between_projections(obj_a, obj_b, col))
            else:
                X[i, col] = (
                    pow(distance_between_projections(obj_a, i, col), 2)
                    + pow(distance_between_projections(obj_a, obj_b, col), 2)
                    - pow(distance_between_projections(obj_b, i, col), 2)
                ) / (2 * distance_between_projections(obj_a, obj_b, col))
    # Recursive call
    Fastmap(k - 1)


def get_weight_matrix(r):
    # Create list with indices ranging from -r to r
    ind = list(range(-r, r + 1, 1))

    # Define an empty dataframe with with index from r to -r and columns from -r to r
    weight_matrix = pd.DataFrame(columns=ind, index=ind[::-1])

    # In a for loop iterate through all indices and fill weight matrix dataframe values with equation 2
    # Define a 2D Gaussian distribution around the central pixel within a square window of size (2r+1)×(2r+1)
    for i in ind:
        for j in ind:
            sigma = (2 * r + 1) / 4
            weight_matrix.at[i, j] = math.exp((-pow(i, 2) - pow(j, 2)) / (2 * pow(sigma, 2)))

    # Return weight matrix
    return weight_matrix


def SA(msi_df, r=3, q=20, k=10, connectivity=None, seed_val=0):
    """
    Spatially-aware clustering.

    :param msi_df: pandas dataframe with (x,y) as multi index and m/z values as columns
    :param r: pixel neighborhood radius (keep this small to avoid out-of-memory errors)
    :param q: Fastmap desired dimension
    :param k: number of clusters
    :param connectivity: include pixel connectivity information during clustering
    :param seed_val: seed the random number generator
    :return: class labels
    """
    global data_matrix  # original input data
    global X  # output with the ith row being the image of the ith object at the end of the algorithm
    global PA  # stores the ids of the pivot objects - one pair per recursive call
    global col  # stores the column of the X array currently being updated

    no_pixels = msi_df.shape[0]
    no_peaks = msi_df.shape[1]

    # #######################
    # # 1. given r, create weights
    # #######################
    weight_matrix_df = get_weight_matrix(r)

    # #######################
    # # 2. map a spectrum into the feature space
    # #######################
    # create empty numpy data array of shape no_pixels x weight_matrix_rows x weight_matrix_cols x no_peaks
    data = np.empty((no_pixels, weight_matrix_df.shape[0], weight_matrix_df.shape[1], no_peaks))

    # list with indices ranging from -r to r
    ind = list(range(-r, r + 1, 1))

    # fill data array by iteratively looping through spectra
    # set pixel counter
    px_idx = 0
    for pixel, _spectra in msi_df.iterrows():
        # create an empty array to save phi(s)
        phi_s = np.empty((weight_matrix_df.shape[0], weight_matrix_df.shape[1], no_peaks))

        # iterate through every index in weight matrix and fill phi(s)
        for i in ind:
            for j in ind:
                # set x and y pixel according to i and j
                x = i + pixel[0]
                y = j + pixel[1]

                # if x and y in dataframe store spectra at x,y in s, otherwise set s as zero vector
                s = msi_df.loc[x, y].to_numpy() if (x, y) in msi_df.index else np.zeros(no_peaks)

                # calculate phi
                phi_s[i, j] = math.sqrt(weight_matrix_df.at[i, j]) * s

        # insert phi(s) into data matrix
        data[px_idx] = phi_s

        # increase pixel counter
        px_idx += 1

    #######################
    # 3. Given q, project mapped spectra into Rq using FastMap
    #######################
    data_matrix = data
    X = np.zeros((data_matrix.shape[0], q))
    PA = np.zeros((2, q))
    col = -1

    random.seed(seed_val)
    Fastmap(q)

    np.set_printoptions(suppress=True)
    # print(X)
    # print(PA)
    # print(X.shape)

    #######################
    # 4. Cluster the projected mapped spectra into k groups using k-means
    #######################
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed_val)
    kmeans.fit(np.float32(X))
    class_labels = kmeans.labels_

    return class_labels
