#Imports
import numpy as np
import numba
from numba import njit
import scipy.io
import os

__all__ = ['compute_norm', '_cross', 'max_edge_length','decimal_to_digits',
                           'float_to_int', 'transform', 'inv_transform','SimpleImplicitSurfaceProjection','read_mesh_data']

_TYPE_MAP = [("f8", "i4"), ("f8", "i8")]
NB_OPTS = {"nogil": True}
NB_OPTS = {"nogil": True}
# which works out to be 1e-13
TOL_ZERO = np.finfo(np.float64).resolution * 100
# how close to merge vertices
TOL_MERGE = 1e-8


@numba.njit(["{0}({0}[:])".format("f8")], **NB_OPTS)
def compute_norm(vec):
    sqnorm = np.float64(0.0)
    for i in range(len(vec)):
        sqnorm += vec[i] * vec[i]
    sqnorm = np.sqrt(sqnorm)
    return sqnorm


@numba.njit(["{0}[:]({0}[:], {0}[:])".format("f8")], **NB_OPTS)
def _cross(a, b):
    r"""Cross product axb

    Parameters
    ----------
    a, b: np.ndarray
        nx3 coordinates

    Returns
    -------
    np.ndarray
        The cross product of :math:`\boldsymbol{a}\times\boldsymbol{b}`.
    """

    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@numba.njit(["{0}({0}[:,:])".format("f8")], **NB_OPTS)
def max_edge_length(xs):
    # compute maximum edge length of elements

    return max(
        compute_norm(xs[0] - xs[1]),
        compute_norm(xs[1] - xs[2]),
        compute_norm(xs[2] - xs[0]),
    )


def decimal_to_digits(decimal, min_digits=None):
    """
    Return the number of digits to the first nonzero decimal.
    Parameters
    -----------
    decimal:    float
    min_digits: int, minimum number of digits to return
    Returns
    -----------
    digits: int, number of digits to the first nonzero decimal
    """
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits


def float_to_int(data, digits=None, dtype=np.int32):
    """
    Given a numpy array of float/bool/int, return as integers.
    Parameters
    -------------
    data :  (n, d) float, int, or bool
      Input data
    digits : float or int
      Precision for float conversion
    dtype : numpy.dtype
      What datatype should result be returned as
    Returns
    -------------
    as_int : (n, d) int
      Data as integers
    """
    # convert to any numpy array
    data = np.asanyarray(data)
    # if data is already an integer or boolean we're done
    # if the data is empty we are also done
    if data.dtype.kind in 'ib' or data.size == 0:
        return data.astype(dtype)
    elif data.dtype.kind != 'f':
        data = data.astype(np.float64)
    # vertices closer than this should be merged
        tol_merge = 1e-8
    # populate digits from kwargs
    if digits is None:
        digits = decimal_to_digits(tol_merge)
    elif isinstance(digits, float) or isinstance(digits, np.float64):
        digits = decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warning('Digits were passed as %s!', digits.__class__.__name__)
        raise ValueError('Digits must be None, int, or float!')
    # data is float so convert to large integers
    data_max = np.abs(data).max() * 10**digits
    # ignore passed dtype if we have something large
    dtype = [np.int32, np.int64][int(data_max > 2**31)]
    # multiply by requested power of ten
    # then subtract small epsilon to avoid "go either way" rounding
    # then do the rounding and convert to integer
    as_int = np.round((data * 10 ** digits) - 1e-6).astype(dtype)

    return as_int

def transform(unisolvent_nodes, duffy_transform=False):
    """
    Transform Chebyshev points from [-1, 1]^2 to a reference simplex.

    Parameters:
        unisolvent_nodes (ndarray): Chebyshev points on the square.
        duffy_transform (bool): Whether to apply Duffy's transform.

    Returns:
        ndarray: Transformed points on the simplex.
    """
    x, y = unisolvent_nodes[:,0],unisolvent_nodes[:,1]
    if duffy_transform:
        points_simplex_x = (1/4) * ((1 + x) * (1 - y))
        points_simplex_y = (1 + y) / 2
    else:
        points_simplex_x = (1 + x) * (3 - y) / 8
        points_simplex_y = (3 - x) * (y + 1) / 8

    return np.column_stack((points_simplex_x, points_simplex_y))


def inv_transform(qpoint_triangle, duffy_transform=False):
    """
    Transform quadrature points from the reference simplex to a unit square.

    Parameters:
        qpoint_triangle (ndarray): Quadrature points on the reference simplex.
        duffy_transform (bool): Whether to apply Duffy's transform.

    Returns:
        ndarray: Transformed points on the [-1, 1]^2.
    """
    x, y = qpoint_triangle[:, 0], qpoint_triangle[:, 1]

    if duffy_transform:
        qpoint_square_x = (2 * x / (1 - y)) - 1
        qpoint_square_y = 2 * y - 1
    else:
        sqrt_term = np.sqrt((x - y) ** 2 + 4 * (1 - x - y))
        qpoint_square_x = 1 + (x - y) - sqrt_term
        qpoint_square_y = 1 - (x - y) - sqrt_term

    return np.column_stack((qpoint_square_x, qpoint_square_y))

 
# @njit(fastmath=True)
def SimpleImplicitSurfaceProjection(phi: callable, dphi: callable, x: np.ndarray, max_iter=10) -> np.ndarray:
    """Closest-point projection to surface given by an implicit function using a simple projection algorithm.
    
    Args:
        phi: Zero-levelset function
        dphi: Gradient of zero-levelset function
        x: The point to be projected
        max_iter: Maximum number of iterations for the projection
        
    Returns:
        The projection point"""

    tol = 10 * np.finfo(np.float64).eps
    phi_v = phi(x)
    for i in range(max_iter):
        grad_phi = dphi(x)
        grad_phi_norm = np.sum(grad_phi**2)
        normalize = phi_v / grad_phi_norm

        if np.sqrt(phi_v * normalize) < tol:
            break

        for j in range(len(x)):
            x[j] -= grad_phi[j] * normalize

        phi_v = phi(x)

    return x

def read_mesh_data(mesh_path):
    """
    Read mesh data from a MAT file.

    Args:
        mesh_path (str): The file path to the MAT file containing mesh data.

    Returns:
        vertices (numpy.ndarray): The 'vertices' data from the MAT file.
        faces (numpy.ndarray): The 'faces' data from the MAT file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs during file reading.
    """
    try:
        # Check if the file exists
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"File not found: {mesh_path}")

        # Load the MAT file
        mesh_mat = scipy.io.loadmat(mesh_path)

        # Get a list of keys in the loaded dictionary
        key_list = list(mesh_mat.keys())

        # Access the 'vertices' and 'faces' data
        vertices = mesh_mat[key_list[-1]]
        faces = mesh_mat[key_list[-2]] - 1  

        return vertices, faces
    except Exception as e:
        print(f"An error occurred while reading the mesh data: {e}")
        return None, None
