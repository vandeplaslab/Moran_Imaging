"""Defining spatial weights matrix."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components

if TYPE_CHECKING:
    from moran_imaging._typing import ContiguityType


def define_spatial_weights_matrix(
    image_shape: tuple[int, int],
    contiguity: ContiguityType = "queen",
    neighbourhood_order: int = 1,
    background_mask: np.ndarray | None = None,
    with_lower_order: bool = False,
):
    """
    Define a Queen or Rook spatial weights contiguity matrix.

    Inputs
    ----------
    image_shape         : tuple
                          Image dimensions (number of rows, number of columns).
    contiguity          : string
                          Type of contiguity. By default, Queen contiguity. Otherwise, Rook contiguity.
    neighbourhood_order : integer
                          Number of contiguous pixels in neighbourhood (order of contiguity). By default, one.
    background_mask     : array
                          Boolean numpy array of missing values of size (one boolean variable per pixel).
    with_lower_order    : boolean
                          True to include lower order contiguities, False otherwise. By default, False.

    Output
    -------
    W : Contiguity-based spatial weights matrix object.
    """
    if background_mask is None:
        background_mask = np.array([])
    background_mask = np.asarray(background_mask)

    # Convert boolean background mask to list of missing value indices
    # Pixels with missing values become isolates/islands of the spatial weights matrix
    background_index = [] if len(background_mask) == 0 else np.argwhere(background_mask)[:, 0].tolist()

    # Define 1st order Queen or Rook contiguity matrix
    contiguity = contiguity.lower()
    if contiguity == "queen":
        W = define_lattice_spatial_weights_matrix(
            image_shape[0], image_shape[1], contiguity="queen", missing=background_index
        )
    elif contiguity == "rook":
        W = define_lattice_spatial_weights_matrix(
            image_shape[0], image_shape[1], contiguity="rook", missing=background_index
        )

    # Compute higher-order neighbourhood
    if neighbourhood_order > 1:
        W = define_higher_order_spatial_weights(W, neighbourhood_order, lower_order=with_lower_order)

    return W


def define_lattice_spatial_weights_matrix(
    nrows: int, ncols: int, contiguity: ContiguityType = "queen", missing: list | None = None
) -> SpatialWeightsMatrix:
    """
    Define a contiguity-based spatial weights matrix for a regular lattice.
    Code adapted from https://pysal.org/libpysal/_modules/libpysal/weights/util.html#lat2W
    & https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W
    Observations are row ordered (first ncols observations are in row 0, next ncols in row 1, next ncols in row 2,
     etc).

    Inputs
    ----------
    nrows      : integer
                 Number of rows in the lattice.
    ncols      : integer
                 Number of columns in the lattice.
    contiguity : ContiguityType
                 Type of contiguity. By default, Queen contiguity. Otherwise, Rook contiguity.
    missing    : list
                 Missing value indices to be excluded from the spatial weights matrix. By default, empty list.

    Output
    -------
    spatial_weights_matrix : First-order contiguity-based spatial weights matrix object.
    """
    if missing is None:
        missing = []
    n = nrows * ncols
    r1 = nrows - 1
    c1 = ncols - 1
    rid = [i // ncols for i in range(n)]  # must be floor!
    cid = [i % ncols for i in range(n)]

    # Use Joblib to parallelize the computation of neighbors
    results = Parallel(n_jobs=-1, prefer="threads", verbose=0, timeout=None)(
        delayed(compute_neighbors)(i, ncols, nrows, r1, c1, rid, cid, contiguity) for i in range(n)
    )

    # Create the spatial weights matrix
    w = dict(enumerate(results))

    # Remove weights corresponding to missing values
    if missing:
        for missing_value_index in missing:
            w[missing_value_index] = []

    weights = {}
    for key in w:
        weights[key] = [1.0] * len(w[key])
    ids = list(range(n))

    return SpatialWeightsMatrix(neighbors=w, weights=weights, ids=ids, id_order=ids[:], silence_warnings=True)


def compute_neighbors(i: int, ncols: int, nrows: int, r1, c1, rid, cid, contiguity):
    """Compute neighbors for a single cell at index i."""
    neighbors = []

    # Rook contiguity (edge neighbors)
    if rid[i] < r1:  # below neighbor
        below = i + ncols
        neighbors.append(below)
    if rid[i] > 0:  # above neighbor
        above = i - ncols
        neighbors.append(above)
    if cid[i] < c1:  # right neighbor
        right = i + 1
        neighbors.append(right)
    if cid[i] > 0:  # left neighbor
        left = i - 1
        neighbors.append(left)

    # Queen contiguity (corner neighbors)
    contiguity = contiguity.lower()
    if contiguity == "queen":
        if rid[i] < r1 and cid[i] < c1:  # bottom-right neighbor
            bottom_right = i + ncols + 1
            neighbors.append(bottom_right)
        if rid[i] < r1 and cid[i] > 0:  # bottom-left neighbor
            bottom_left = i + ncols - 1
            neighbors.append(bottom_left)
        if rid[i] > 0 and cid[i] < c1:  # top-right neighbor
            top_right = i - ncols + 1
            neighbors.append(top_right)
        if rid[i] > 0 and cid[i] > 0:  # top-left neighbor
            top_left = i - ncols - 1
            neighbors.append(top_left)
    return neighbors


def define_higher_order_spatial_weights(w, k: int, lower_order: bool) -> SpatialWeightsMatrix:
    """
    Define a contiguity-based spatial weights matrix for a neighbourhood order larger or equal than two pixels.
    From https://pysal.org/libpysal/_modules/libpysal/weights/util.html#higher_order.

    Inputs
    ----------
    w             : First-order contiguity-based spatial weights object
    k             : integer
                    Order of contiguity
    lower_order   : boolean
                    True to include lower order contiguities ; False to return only weights of order k.

    Outputs
    -------
    spatial_weights_matrix : Higher-order contiguity-based spatial weights matrix object.
    """
    id_order = w.id_order
    shortest_path = True
    w = w.sparse

    if lower_order:
        wk = sum(w**x for x in range(2, k + 1))
        shortest_path = False
    else:
        wk = w**k

    rk, ck = wk.nonzero()
    sk = set(zip(rk, ck))

    if shortest_path:
        for j in range(1, k):
            wj = w**j
            rj, cj = wj.nonzero()
            sj = set(zip(rj, cj))
            sk.difference_update(sj)

    sk = {(i, j) for i, j in sk if i != j}
    d = {i: [] for i in id_order}

    for pair in sk:
        k, v = pair
        k = id_order[k]
        v = id_order[v]
        d[k].append(v)

    return SpatialWeightsMatrix(neighbors=d, silence_warnings=True)


class SpatialWeightsMatrix:
    r"""
    Spatial weights matrix class.

    Parameters
    ----------
    neighbors : dict
        Key is region ID, value is a list of neighbor IDS.
        For example, {'a':['b'],'b':['a','c'],'c':['b']}.
    weights : dict
       Key is region ID, value is a list of edge weights.
       If not supplied all edge weights are assumed to have a weight of 1.
       For example, {'a':[0.5],'b':[0.5,1.5],'c':[1.5]}.
    id_order : list
       An ordered list of ids, defines the order of observations when
       iterating over W if not set, lexicographical ordering is used
       to iterate and the id_order_set property will return False.
       This can be set after creation by setting the id_order property.
    silence_warnings : bool
       Warning in case of islands. Silence warning by setting to True
    ids : list
        Values to use for keys of the neighbors and weights dict objects.

    Attributes
    ----------
    asymmetries: List of id pairs with asymmetric weights.
    cardinalities: Number of neighbors for each observation.
    component_labels: Store the graph component in which each observation falls.
    diagW2: Diagonal of WW.
    diagWtW: Diagonal of W'W.
    diagWtW_WW: Diagonal of W'W + WW.
    histogram: Cardinality histogram as a dictionary where key is the id and value is the number of neighbors for
     that unit.
    id2i: Dictionary where the key is an ID and the value is that ID's index in W.id_order.
    id_order: Returns the ids for the observations in the order in which they would be encountered if iterating over
     the weights.
    id_order_set: Returns True if user has set id_order, False if not.
    islands: List of ids without any neighbors.
    max_neighbors: Largest number of neighbors.
    mean_neighbors: Average number of neighbors.
    min_neighbors: Minimum number of neighbors.
    n: Number of units.
    n_components: Store whether the adjacency matrix is fully connected.
    neighbor_offsets: Given the current id_order, neighbor_offsets[id] is the offsets of the id's neighbors in
     id_order.
    nonzero: Number of nonzero weights.
    pct_nonzero: Percentage of nonzero weights.
    s0: s0=\sum_i \sum_j w_{i,j}
    s1: s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2
    s2: s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2
    s2array: Individual elements comprising s2.
    sd: Standard deviation of number of neighbors.
    sparse: Sparse matrix object.
    trcW2: Trace of WW
    trcWtW: Trace of W'W
    trcWtW_WW: Trace of W'W + WW
    transform: Getter for transform property.

    Methods
    -------
    asymmetry([intrinsic]): Asymmetry check.
    full(): Generate a full numpy.ndarray.
    get_transform(): Getter for transform property.
    remap_ids(new_ids): In place modification throughout W of id values from w.id_order to new_ids in all.
    set_transform([value]): Transformation of the weights.
    """

    def __init__(
        self,
        neighbors: dict,
        weights: dict | None = None,
        id_order: list | None = None,
        silence_warnings: bool = False,
        ids: list | None = None,
    ):
        self.silence_warnings = silence_warnings
        self.transformations = {}
        self.neighbors = neighbors
        if not weights:
            weights = {}
            for key in neighbors:
                weights[key] = [1.0] * len(neighbors[key])
        self.weights = weights
        self.transformations["O"] = self.weights.copy()  # original weights
        self.transform = "O"
        if id_order is None:
            self._id_order = list(self.neighbors.keys())
            self._id_order.sort()
            self._id_order_set = False
        else:
            self._id_order = id_order
            self._id_order_set = True
        self._reset()
        self._n = len(self.weights)
        if (not self.silence_warnings) and (self.n_components > 1):
            message = (
                "The weights matrix is not fully connected: \n There are %d disconnected components."
                % self.n_components
            )
            ni = len(self.islands)
            if ni == 1:
                message = message + "\n There is 1 island with id: %s." % (str(self.islands[0]))
            elif ni > 1:
                message = message + "\n There are %d islands with ids: %s." % (
                    ni,
                    ", ".join(str(island) for island in self.islands),
                )
            warnings.warn(message)

    def _reset(self) -> None:
        """Reset properties."""
        self._cache = {}

    @property
    def sparse(self):
        """
        Sparse matrix object. For any matrix manipulations required for w, w.sparse should be used.

        This is based on scipy.sparse.
        """
        if "sparse" not in self._cache:
            self._sparse = self._build_sparse()
            self._cache["sparse"] = self._sparse
        return self._sparse

    @property
    def n_components(self):
        """Store whether the adjacency matrix is fully connected."""
        if "n_components" not in self._cache:
            self._n_components, self._component_labels = connected_components(self.sparse)
            self._cache["n_components"] = self._n_components
            self._cache["component_labels"] = self._component_labels
        return self._n_components

    @property
    def component_labels(self):
        """Store the graph component in which each observation falls."""
        if "component_labels" not in self._cache:
            self._n_components, self._component_labels = connected_components(self.sparse)
            self._cache["n_components"] = self._n_components
            self._cache["component_labels"] = self._component_labels
        return self._component_labels

    def _build_sparse(self):
        """Construct the sparse attribute."""
        row = []
        col = []
        data = []
        id2i = self.id2i
        for i, neigh_list in list(self.neighbor_offsets.items()):
            card = self.cardinalities[i]
            row.extend([id2i[i]] * card)
            col.extend(neigh_list)
            data.extend(self.weights[i])
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        s = sp.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        return s

    @property
    def id2i(self):
        """Dictionary where the key is an ID and the value is that ID's index in W.id_order."""
        if "id2i" not in self._cache:
            self._id2i = {}
            for i, id_i in enumerate(self._id_order):
                self._id2i[id_i] = i
            self._id2i = self._id2i
            self._cache["id2i"] = self._id2i
        return self._id2i

    @property
    def n(self):
        """Number of units."""
        if "n" not in self._cache:
            self._n = len(self.neighbors)
            self._cache["n"] = self._n
        return self._n

    @property
    def s0(self):
        r"""s0 is defined as
        .. math::
               s0=\sum_i \sum_j w_{i,j}.
        """
        if "s0" not in self._cache:
            self._s0 = self.sparse.sum()
            self._cache["s0"] = self._s0
        return self._s0

    @property
    def s1(self):
        r"""s1 is defined as
        .. math::
               s1=1/2 \sum_i \sum_j \Big(w_{i,j} + w_{j,i}\Big)^2.
        """
        if "s1" not in self._cache:
            t = self.sparse.transpose()
            t = t + self.sparse
            t2 = t.multiply(t)  # element-wise square
            self._s1 = t2.sum() / 2.0
            self._cache["s1"] = self._s1
        return self._s1

    @property
    def s2array(self):
        """Individual elements comprising s2.

        See Also
        --------
        s2

        """
        if "s2array" not in self._cache:
            s = self.sparse
            self._s2array = np.array(s.sum(1) + s.sum(0).transpose()) ** 2
            self._cache["s2array"] = self._s2array
        return self._s2array

    @property
    def s2(self):
        r"""s2 is defined as
        .. math::
                s2=\sum_j \Big(\sum_i w_{i,j} + \sum_i w_{j,i}\Big)^2.
        """
        if "s2" not in self._cache:
            self._s2 = self.s2array.sum()
            self._cache["s2"] = self._s2
        return self._s2

    @property
    def trcW2(self):
        """Trace of :math:`WW`.

        See Also
        --------
        diagW2
        """
        if "trcW2" not in self._cache:
            self._trcW2 = self.diagW2.sum()
            self._cache["trcw2"] = self._trcW2
        return self._trcW2

    @property
    def diagW2(self):
        """Diagonal of :math:`WW`.

        See Also
        --------
        trcW2
        """
        if "diagw2" not in self._cache:
            self._diagW2 = (self.sparse * self.sparse).diagonal()
            self._cache["diagW2"] = self._diagW2
        return self._diagW2

    @property
    def diagWtW(self):
        """Diagonal of :math:`W^{'}W`.

        See Also
        --------
        trcWtW
        """
        if "diagWtW" not in self._cache:
            self._diagWtW = (self.sparse.transpose() * self.sparse).diagonal()
            self._cache["diagWtW"] = self._diagWtW
        return self._diagWtW

    @property
    def trcWtW(self):
        """Trace of :math:`W^{'}W`.

        See Also
        --------
        diagWtW
        """
        if "trcWtW" not in self._cache:
            self._trcWtW = self.diagWtW.sum()
            self._cache["trcWtW"] = self._trcWtW
        return self._trcWtW

    @property
    def diagWtW_WW(self):
        """Diagonal of :math:`W^{'}W + WW`."""
        if "diagWtW_WW" not in self._cache:
            wt = self.sparse.transpose()
            w = self.sparse
            self._diagWtW_WW = (wt * w + w * w).diagonal()
            self._cache["diagWtW_WW"] = self._diagWtW_WW
        return self._diagWtW_WW

    @property
    def trcWtW_WW(self):
        """Trace of :math:`W^{'}W + WW`."""
        if "trcWtW_WW" not in self._cache:
            self._trcWtW_WW = self.diagWtW_WW.sum()
            self._cache["trcWtW_WW"] = self._trcWtW_WW
        return self._trcWtW_WW

    @property
    def pct_nonzero(self):
        """Percentage of nonzero weights."""
        if "pct_nonzero" not in self._cache:
            self._pct_nonzero = 100.0 * self.sparse.nnz / (1.0 * self._n**2)
            self._cache["pct_nonzero"] = self._pct_nonzero
        return self._pct_nonzero

    @property
    def cardinalities(self):
        """Number of neighbors for each observation."""
        if "cardinalities" not in self._cache:
            c = {}
            for i in self._id_order:
                c[i] = len(self.neighbors[i])
            self._cardinalities = c
            self._cache["cardinalities"] = self._cardinalities
        return self._cardinalities

    @property
    def max_neighbors(self):
        """Largest number of neighbors."""
        if "max_neighbors" not in self._cache:
            self._max_neighbors = max(self.cardinalities.values())
            self._cache["max_neighbors"] = self._max_neighbors
        return self._max_neighbors

    @property
    def mean_neighbors(self):
        """Average number of neighbors."""
        if "mean_neighbors" not in self._cache:
            self._mean_neighbors = np.mean(list(self.cardinalities.values()))
            self._cache["mean_neighbors"] = self._mean_neighbors
        return self._mean_neighbors

    @property
    def min_neighbors(self):
        """Minimum number of neighbors."""
        if "min_neighbors" not in self._cache:
            self._min_neighbors = min(self.cardinalities.values())
            self._cache["min_neighbors"] = self._min_neighbors
        return self._min_neighbors

    @property
    def nonzero(self):
        """Number of nonzero weights."""
        if "nonzero" not in self._cache:
            self._nonzero = self.sparse.nnz
            self._cache["nonzero"] = self._nonzero
        return self._nonzero

    @property
    def sd(self):
        """Standard deviation of number of neighbors."""
        if "sd" not in self._cache:
            self._sd = np.std(list(self.cardinalities.values()))
            self._cache["sd"] = self._sd
        return self._sd

    @property
    def asymmetries(self):
        """List of id pairs with asymmetric weights."""
        if "asymmetries" not in self._cache:
            self._asymmetries = self.asymmetry()
            self._cache["asymmetries"] = self._asymmetries
        return self._asymmetries

    @property
    def islands(self):
        """List of ids without any neighbors."""
        if "islands" not in self._cache:
            self._islands = [i for i, c in list(self.cardinalities.items()) if c == 0]
            self._cache["islands"] = self._islands
        return self._islands

    @property
    def histogram(self):
        """Cardinality histogram as a dictionary where key is the id and value is the number of neighbors for that unit."""
        if "histogram" not in self._cache:
            ct, bin = np.histogram(
                list(self.cardinalities.values()),
                list(range(self.min_neighbors, self.max_neighbors + 2)),
            )
            self._histogram = list(zip(bin, ct))
            self._cache["histogram"] = self._histogram
        return self._histogram

    def __getitem__(self, key):
        """Allow a dictionary like interaction with the weights class."""
        return dict(list(zip(self.neighbors[key], self.weights[key])))

    def __iter__(self):
        """Support iteration over weights."""
        for i in self._id_order:
            yield i, dict(list(zip(self.neighbors[i], self.weights[i])))

    def remap_ids(self, new_ids):
        """
        In place modification throughout W of id values from w.id_order to new_ids in all.

        Parameters
        ----------
        new_ids : list, numpy.ndarray
            Aligned list of new ids to be inserted. Note that first element of new_ids will replace first element of w.id_order,
            second element of new_ids replaces second element of w.id_order and so on.
        """
        old_ids = self._id_order
        if len(old_ids) != len(new_ids):
            raise Exception("W.remap_ids: length of `old_ids` does not match that of new_ids")
        if len(set(new_ids)) != len(new_ids):
            raise Exception("W.remap_ids: list `new_ids` contains duplicates")
        else:
            new_neighbors = {}
            new_weights = {}
            old_transformations = self.transformations["O"].copy()
            new_transformations = {}
            for o, n in zip(old_ids, new_ids):
                o_neighbors = self.neighbors[o]
                o_weights = self.weights[o]
                n_neighbors = [new_ids[old_ids.index(j)] for j in o_neighbors]
                new_neighbors[n] = n_neighbors
                new_weights[n] = o_weights[:]
                new_transformations[n] = old_transformations[o]
            self.neighbors = new_neighbors
            self.weights = new_weights
            self.transformations["O"] = new_transformations

            id_order = [self._id_order.index(o) for o in old_ids]
            for i, id_ in enumerate(id_order):
                self.id_order[id_] = new_ids[i]

            self._reset()

    def __set_id_order(self, ordered_ids):
        """Set the iteration order in w. W can be iterated over.
        On construction the iteration order is set to the lexicographic order of the keys in the w.weights dictionary.
        If a specific order is required it can be set with this method.

        Parameters
        ----------
        ordered_ids : sequence
            Identifiers for observations in specified order.

        Notes
        -----
        The ordered_ids parameter is checked against the ids implied by the keys in w.weights.
        If they are not equivalent sets an exception is raised and the iteration order is not changed.
        """
        if set(self._id_order) == set(ordered_ids):
            self._id_order = ordered_ids
            self._id_order_set = True
            self._reset()
        else:
            raise Exception("ordered_ids do not align with W ids")

    def __get_id_order(self):
        """Retyrb the ids for the observations in the order in which they would be encountered if iterating over the weights."""
        return self._id_order

    id_order = property(__get_id_order, __set_id_order)

    @property
    def id_order_set(self):
        """Returns True if user has set id_order, False if not."""
        return self._id_order_set

    @property
    def neighbor_offsets(self):
        """Given the current id_order, neighbor_offsets[id] is the offsets of the id's neighbors in id_order.

        Returns
        -------
        neighbor_list : list
            Offsets of the id's neighbors in id_order.
        """
        if "neighbors_0" not in self._cache:
            self.__neighbors_0 = {}
            id2i = self.id2i
            for j, neigh_list in list(self.neighbors.items()):
                self.__neighbors_0[j] = [id2i[neigh] for neigh in neigh_list]
            self._cache["neighbors_0"] = self.__neighbors_0
        return self.__neighbors_0

    def get_transform(self):
        """Getter for transform property.

        Returns
        -------
        transformation : str, None
            Valid transformation value. See the transform parameters in set_transform() for a detailed description.

        See Also
        --------
        set_transform
        """
        return self._transform

    def set_transform(self, value="B"):
        """Transformations of weights.

        Parameters
        ----------
        transform : str
            This parameter is not case sensitive. The following are
            valid transformations.

            * **B** -- Binary
            * **R** -- Row-standardization (global sum :math:`=n`)
            * **D** -- Double-standardization (global sum :math:`=1`)
            * **O** -- Restore original transformation (from instantiation)

        Notes
        -----
        Transformations are applied only to the value of the weights at instantiation.
        Chaining of transformations cannot be done on a W instance.
        """
        value = value.upper()
        self._transform = value
        if value in self.transformations:
            self.weights = self.transformations[value]
            self._reset()
        else:
            if value == "R":
                # row standardized weights
                weights = {}
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    row_sum = sum(wijs) * 1.0
                    if row_sum == 0.0 and not self.silence_warnings:
                        pass
                    weights[i] = [wij / row_sum for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "D":
                # doubly-standardized weights
                # update current chars before doing global sum
                self._reset()
                s0 = self.s0
                ws = 1.0 / s0
                weights = {}
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i] = [wij * ws for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "B":
                # binary transformation
                weights = {}
                self.weights = self.transformations["O"]
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i] = [1.0 for wij in wijs]
                weights = weights
                self.transformations[value] = weights
                self.weights = weights
                self._reset()
            elif value == "O":
                # put weights back to original transformation
                weights = {}
                original = self.transformations[value]
                self.weights = original
                self._reset()
            else:
                raise Exception("unsupported weights transformation")

    transform = property(get_transform, set_transform)

    def asymmetry(self, intrinsic=True):
        """Asymmetry check. Intrinsic symmetry is defined as w_{i,j} == w_{j,i}.

        Parameters
        ----------
        intrinsic : bool
            Default is True.

        Returns
        -------
        asymmetries : list
            Empty if no asymmetries are found if asymmetries, then a
            list of (i,j) tuples is returned.
        """
        if intrinsic:
            wd = self.sparse.transpose() - self.sparse
        else:
            transform = self.transform
            self.transform = "b"
            wd = self.sparse.transpose() - self.sparse
            self.transform = transform

        ids = np.nonzero(wd)
        if len(ids[0]) == 0:
            return []
        ijs = list(zip(ids[0], ids[1]))
        ijs.sort()
        return ijs

    def full(self):
        """Generate a full numpy.ndarray of the spatial weights matrix.

        Parameters
        ----------
        self : libpysal.weights.W
            spatial weights object

        Returns
        -------
        (fullw, keys) : tuple
            The first element being the full numpy.ndarray and second
            element keys being the ids associated with each row in the array.
        """
        wfull = self.sparse.toarray()
        keys = list(self.neighbors.keys())
        if self.id_order:
            keys = self.id_order
        return (wfull, keys)

    def to_adjlist(self, focal_col="focal", neighbor_col="neighbor", weight_col="weight"):
        """
        Compute an adjacency list representation of the spatial weights matrix.
        Symmetric entries are considered as distinct links, with potentially distinct weights.
        Islands (pixels without any neighbors) do not count, and do not appear in the adjacency list.

        Parameters
        ----------
        focal_col : str
            Name of the column in which to store "source" node ids.
        neighbor_col : str
            Name of the column in which to store "destination" node ids.
        weight_col : str
            Name of the column in which to store weight information.
        """
        focal_ix, neighbor_ix = self.sparse.nonzero()
        names = np.asarray(self.id_order)
        focal = names[focal_ix]
        neighbor = names[neighbor_ix]
        weights = self.sparse.data
        adjlist = pandas.DataFrame({focal_col: focal, neighbor_col: neighbor, weight_col: weights})

        island_adjlist = pandas.DataFrame({focal_col: self.islands, neighbor_col: self.islands, weight_col: 0})
        adjlist = pandas.concat((adjlist, island_adjlist)).reset_index(drop=True)

        return adjlist.sort_values([focal_col, neighbor_col])


spatial_weights_matrix = SpatialWeightsMatrix
