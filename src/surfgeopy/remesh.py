#Imports
import numpy as np

#Local Imports
from .utils import *
"""
remesh.py
-------------
Deal with re- triangulation of existing meshes.
"""
__all__ = ['subdivide','faces_to_edges','unique_rows', 'hashable_rows', 'unique_ordered']


def subdivide(vertices,
              faces,
              face_index=None,
              vertex_attributes=None,
              return_index=False):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = unique_rows(edges)
    
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            attr_mid = np.vstack([
                attr_tris[:, g, :].mean(axis=1)
                for g in [[0, 1],
                          [1, 2],
                          [2, 0]]])
            attr_mid = attr_mid[unique]
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        
        stack = np.arange(
            start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = {k: v for k, v in zip(nonzero, stack)}

        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces



def faces_to_edges(faces, return_index=False):
    """
    Given a list of faces (n,3), return a list of edges (n*3,2)
    Parameters
    -----------
    faces : (n, 3) int
      Vertex indices representing faces
    Returns
    -----------
    edges : (n*3, 2) int
      Vertex indices representing edges
    """
    faces = np.asanyarray(faces)

    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    if return_index:
        # edges are in order of faces due to reshape
        face_index = np.tile(np.arange(len(faces)),
                             (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges



def unique_rows(data, digits=None, keep_order=False):
    """
    Returns indices of unique rows. It will return the
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]
    
    Parameters
    ---------
    data : (n, m) array
      Floating point data
    digits : int or None
      How many digits to consider
    Returns
    --------
    unique :  (j,) int
      Index in data which is a unique row
    inverse : (n,) int
      Array to reconstruct original
      Example: data[unique][inverse] == data
    """
    # get rows hashable so we can run unique function on it
    rows = hashable_rows(data, digits=digits)

    # we are throwing away the first value which is the
    # garbage row-hash and only returning index and inverse
    if keep_order:
        # keeps order of original occurrence
        return unique_ordered(
            rows, return_index=True, return_inverse=True)[1:]
    # returns values sorted by row-hash but since our row-hash
    # were pretty much garbage the sort order isn't meaningful
    return np.unique(rows, return_index=True, return_inverse=True)[1:]


def hashable_rows(data, digits=None):
    """
    We turn our array into integers based on the precision
    given by digits and then put them in a hashable format.
    Parameters
    ---------
    data : (n, m) array
      Input data
    digits : int or None
      How many digits to add to hash if data is floating point
      If None, tol.merge will be used
    Returns
    ---------
    hashable : (n,) array
      Custom data type which can be sorted
      or used as hash keys
    """
    # if there is no data return immediately
    if len(data) == 0:
        return np.array([])

    # get array as integer to precision we care about
    as_int = float_to_int(data, digits=digits)

    # if it is flat integers already, return
    if len(as_int.shape) == 1:
        return as_int

    # if array is 2D and smallish, we can try bitbanging
    # this is significantly faster than the custom dtype
    if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
        # time for some righteous bitbanging
        # can we pack the whole row into a single 64 bit integer
        precision = int(np.floor(64 / as_int.shape[1]))
        # if the max value is less than precision we can do this
        if np.abs(as_int).max() < 2**(precision - 1):
            # the resulting package
            hashable = np.zeros(len(as_int), dtype=np.int64)
            # loop through each column and bitwise xor to combine
            # make sure as_int is int64 otherwise bit offset won't work
            for offset, column in enumerate(as_int.astype(np.int64).T):
                # will modify hashable in place
                np.bitwise_xor(hashable,
                               column << (offset * precision),
                               out=hashable)
            return hashable

    # reshape array into magical data type that is weird but hashable
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    # make sure result is contiguous and flat
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    return hashable


def unique_ordered(data, return_index=False, return_inverse=False):
    """
    Returns the same as np.unique, but ordered as per the
    first occurrence of the unique value in data.
    Examples
    ---------
    In [1]: a = [0, 3, 3, 4, 1, 3, 0, 3, 2, 1]
    In [2]: np.unique(a)
    Out[2]: array([0, 1, 2, 3, 4])
    In [3]: trimesh.grouping.unique_ordered(a)
    Out[3]: array([0, 3, 4, 1, 2])
    """
    # uniques are the values, sorted
    # index is the value in the original `data`
    # i.e. `data[index] == unique`
    # inverse is how to re-construct `data` from `unique`
    # i.e. `unique[inverse] == data`
    unique, index, inverse = np.unique(
        data, return_index=True, return_inverse=True)

    # we want to maintain the original index order
    order = index.argsort()

    if not return_index and not return_inverse:
        return unique[order]

    # collect return values
    # start with the unique values in original order
    result = [unique[order]]
    # the new index values
    if return_index:
        # re-order the index in the original array
        result.append(index[order])
    if return_inverse:
        # create the new inverse from the order of the order
        result.append(order.argsort()[inverse])

    return result
