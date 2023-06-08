#Imports
import numpy as np
import numba

__all__ = ['compute_norm', '_cross', 'max_edge_length','decimal_to_digits',
                           'float_to_int', 'transform', 'inv_transform']

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


def transform(unisolvent_nodes):
    """
    Transform Chebyshev points from [-1,1]^2 to a reference simplex.
    Parameters
    -------------
      Input data
      Chebyshev points on square

    Returns
    -------------
    Chebyshev points on the simplex
    """
    points_simplex = np.zeros((unisolvent_nodes.shape[0], 2))
    points_simplex[:, 0] = 1 / 8 * \
        ((1 + unisolvent_nodes[:, 0]) * (3 - unisolvent_nodes[:, 1]))
    points_simplex[:, 1] = 1 / 8 * \
        ((3 - unisolvent_nodes[:, 0]) * (unisolvent_nodes[:, 1] + 1))
    
     #duffy's transform
#     points_simplex[:,0]=1/4*((1+unisolvent_nodes[:,0])*(1-unisolvent_nodes[:,1]))
#     points_simplex[:,1]=((1+unisolvent_nodes[:,1]))/2

    return points_simplex


def inv_transform(qpoint_triangle):
    """
    Transform quadrature points from the reference simplex to a unit square
    Parameters
    -------------
      Input data
      Quadrature points on the reference simplex

    Returns
    -------------
    Quadrature points on the unit square
    """
    qpoint_square = np.zeros((qpoint_triangle.shape[0], 2))
    qpoint_square[:, 0] = 1 + (qpoint_triangle[:, 0] - qpoint_triangle[:, 1]) - np.sqrt(
        (qpoint_triangle[:, 0] - qpoint_triangle[:, 1])**2 + 4 * (1 - qpoint_triangle[:, 0] - qpoint_triangle[:, 1]))
    qpoint_square[:, 1] = 1 - (qpoint_triangle[:, 0] - qpoint_triangle[:, 1]) - np.sqrt(
        (qpoint_triangle[:, 0] - qpoint_triangle[:, 1])**2 + 4 * (1 - qpoint_triangle[:, 0] - qpoint_triangle[:, 1]))
    
#     qpoint_square[:,0]=(2*qpoint_triangle[:,0]/(1-qpoint_triangle[:,1]))-1
#     qpoint_square[:,1]=2*qpoint_triangle[:,1]-1
    return qpoint_square
