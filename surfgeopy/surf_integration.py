# Imports
import numpy as np
from minterpy import MultiIndexSet, Grid, NewtonPolynomial
from minterpy.dds import dds
from numba import njit
# Local imports
from .quadrature_points import quadrule_on_simplex
from .quadrature_points_gl import gauss_legendre_square
from .remesh import subdivide
from .utils import (
    SimpleImplicitSurfaceProjection, compute_norm, read_mesh_data,
    pushforward, pullback, _cross
)

__all__ = ['integration', 'compute_surf_quadrature', 'quadrature_surf_tri', 'quadrature_split_surf_tri']

def integration(
    ls_function, ls_grad_func, mesh, interp_deg, lp_dgr, Refinement,
    fun_handle=lambda _: 1.0, deg_integration=-1, quadrature_rule=None):
    """
    Compute integration of a function over curved triangles.

    Args:
        ls_function: callable
            Zero-levelset function.
        ls_grad_func: callable
            Gradient of zero-levelset function.
        mesh (str):
            The file path to the MAT file containing mesh data.
        interp_deg: int
            Interpolation degree.
        lp_dgr: int
            :math:`l_p`-norm, which is used to define the polynomial degree.
        Refinement: int
            Refinement level.
        fun_handle: callable, optional
            Function to be evaluated on each quadrature point (default: lambda _: 1.0).
        deg_integration: int, optional
            Degree of integration (default: -1, use default configuration).
        quadrature_rule: str, optional
            Quadrature rule type ('Gauss_Legendre' or 'Gauss_Simplex').

    Returns:
        Integration values for each curved triangle.
    """
    vertices, faces = read_mesh_data(mesh)


    # Case 1: User specifies deg_integration, and quadrature_rule 
    if deg_integration > 0 and quadrature_rule=='Gauss_Legendre':
        pnts, ws, offset = compute_surf_quadrature(
            ls_function, ls_grad_func, vertices, faces,
            interp_deg, lp_dgr, Refinement, fun_handle, deg_integration, quadrature_rule
        )
    # Case 2: User specifies deg_integration and quadrature_rule is default
    elif deg_integration > 0 and quadrature_rule is None:
        pnts, ws, offset = compute_surf_quadrature(
            ls_function, ls_grad_func, vertices, faces,
            interp_deg, lp_dgr, Refinement, fun_handle, deg_integration, quadrature_rule='Pull_back_Gauss'
        )
    else:
        # Default case: Use default configuration if no conditions are met
        pnts, ws, offset = compute_surf_quadrature(
            ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement, fun_handle
        )

    # Perform a function evaluation on each curved triangle
    n_faces = faces.shape[0]
    fs = [0] * n_faces

    for fun_id in range(n_faces):
        for pid in range(offset[fun_id], offset[fun_id + 1]):
            fs[fun_id] += fun_handle(pnts[pid]) * ws[pid]

    return fs



def compute_surf_quadrature(ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement, fun_handle,
                              deg_integration=14,quadrature_rule='Pull_back_Gauss'):
    """
    Compute quadrature points and weights on curved triangles.

    Args:
        ls_function: callable
            Zero-levelset function.
        ls_grad_func: callable
            Gradient of zero-levelset function.
        vertices: ndarray
            Array of vertex coordinates.
        faces: ndarray
            Array of face connectivity.
        interp_deg: int
            Interpolation degree.
        lp_dgr: int
            :math:`l_p`-norm, which is used to define the polynomial degree.
        deg_integration: int, optional
            Degree of integration (default: 14).
        quadrature_rule: str, optional
            Quadrature rule type ('Gauss_Legendre' or 'Gauss_Simplex').

    Returns:
        ndarray, ndarray, ndarray
        Quadrature points, weights, and offset array.
    """

    # Initialization
    index = 0
    n_faces = faces.shape[0]
    nv_surf = faces.shape[1]
    max_nv = max(1000000, n_faces * 6)
    pnts = np.array([[0.0] * 3 for pid in range(max_nv)])
    ws = np.array([0.0 for pid in range(max_nv)])
    offset = np.array([0 for fun_id in range(n_faces + 1)])

    # Go through all the faces
    for fun_id in range(n_faces):
        offset[fun_id] = index

        n_elem = nv_surf - 1
        while faces[fun_id, n_elem] < 0:
            n_elem -= 1
        if n_elem < 2:
            continue

        # Split each element into several curved triangles
        for j in range(1, n_elem):
            lvids = [0, j, j + 1]
            pnts_tri = vertices[faces[fun_id, lvids]]

            # Generate quadrature points
            if Refinement > 0:
                index = quadrature_split_surf_tri(pnts_tri, np.array([[0, 1, 2]]), ls_function, ls_grad_func,
                                                  interp_deg, lp_dgr, Refinement, fun_handle,deg_integration,quadrature_rule, pnts, ws, index)
            else:
                index = quadrature_surf_tri(ls_function, ls_grad_func,
                                            pnts_tri, np.array([[0, 1, 2]]), interp_deg, lp_dgr, fun_handle,deg_integration,quadrature_rule, pnts, ws, index)

    pnts = pnts[:index]
    ws = ws[:index]
    offset[n_faces] = index
    return pnts, ws, offset
def quadrature_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                        lp_dgr, fun_handle, deg_integration, quadrature_rule, pnts, ws, index):
    """
    For a mixed mesh, find the cell integration of the test function f.

    Args:
        ls_function: callable
            Zero-levelset function.
        ls_grad_func: callable
            Gradient of zero-levelset function.
        vertices: ndarray
            Array of vertex coordinates.
        faces: ndarray
            Array of face connectivity.
        interp_deg: int
            Interpolation degree.
        lp_dgr: int
            :math:`l_p`-norm, which is used to define the polynomial degree.
        deg_integration: int
            Degree of integration.
        quadrature_rule: str, optional
            Quadrature rule type ('Gauss_Legendre' or 'Gauss_Simplex').
        pnts: ndarray
            Quadrature points array.
        ws: ndarray
            Quadrature weights array.
        index: int
            Current index in the arrays.

    Returns:
        int
            Updated index value.
    """

    n_faces = faces.shape[0]
    n = interp_deg
    lp = lp_dgr
    pnts_q = np.zeros((1, 3), dtype=np.float64)
    pnts_qq = np.zeros((1, 3), dtype=np.float64)
    mi = MultiIndexSet.from_degree(
        spatial_dimension=2, poly_degree=n, lp_degree=lp)
    grid = Grid(mi)

    # Transform Chebyshev points from [-1,1]^2 to the reference simplex.
    generating_points = pushforward(grid.unisolvent_nodes, duffy_transform=False)
    quad_ps = np.array([[(1.0 - generating_points[row1, 0] - generating_points[row1, 1]),
                         generating_points[row1, 0], generating_points[row1, 1]] for row1 in range(generating_points.shape[0])])
    
    if quadrature_rule == 'Pull_back_Gauss':
        ws0, cs0 = quadrule_on_simplex(deg_integration)
        nqp = ws0.shape[0]
        # Transform quadrature points from the reference simplex to a unit square
        ksi = pullback(cs0, duffy_transform=False)
    else:
        ws0, cs0 = gauss_legendre_square(deg_integration)
        nqp = ws0.shape[0]
    
    # enlarge the size of quadrature points buffer if inadequate
    if index + n_faces * nqp > len(ws):
        n_new = 2 * len(ws) + n_faces * nqp
        ws.resize(n_new, refcheck=False)
        pnts.resize((n_new, 3), refcheck=False)
    for fun_id in range(n_faces):
        pnts_p = np.array([[0.0] * 3 for pid in range(grid.unisolvent_nodes.shape[0])])
        pnts_func = np.array([[0.0] * 1 for pid in range(grid.unisolvent_nodes.shape[0])])
        for q in range(quad_ps.shape[0]):
            pnts_qq = (
                quad_ps[q, 0] * vertices[faces[fun_id, 0]]
                + quad_ps[q, 1] * vertices[faces[fun_id, 1]]
                + quad_ps[q, 2] * vertices[faces[fun_id, 2]])

            pnts_p[q] = SimpleImplicitSurfaceProjection(
                ls_function, ls_grad_func, pnts_qq)
            pnts_func[q]=fun_handle(pnts_p[q])

        interpol_coeffs =np.squeeze(dds(pnts_p, grid.tree))
        newt_poly  = NewtonPolynomial(mi, interpol_coeffs)

  
        # compute partial derivatives with respect to "s"
        ds_poly =newt_poly.diff([1, 0])

        # compute partial derivatives with respect to "t"
        dt_poly =newt_poly.diff([0, 1])
        if quadrature_rule == 'Pull_back_Gauss':
            for qq in range(nqp):
                # there are two alternatives to project quadrature points on the  manifold
                # 1. via closest point projection
                # pnts[index] = SimpleImplicitSurfaceProjection(ls_function, ls_grad_func,pnts_q)
                # 2.  by evaluating the polynomial, which is less exepnsive
                pnts[index] = newt_poly(np.array([ksi[qq, 0], ksi[qq, 1]]))

                # evaluate ∂_s at the quadrature points
                p_s = ds_poly(np.array([ksi[qq, 0], ksi[qq, 1]]))

                # evaluate ∂_t at the quadrature points
                p_t = dt_poly(np.array([ksi[qq, 0], ksi[qq, 1]]))

                # Compute ||∂_s x ∂_t||
                J = compute_norm(_cross(p_s, p_t))
                # Please use this in the case you are applying Duffy' transform
                #ws[index] = ws0[qq] * J * (4/(1-cs0[qq, 1]))
                ws[index] = ws0[qq] * J * \
                   (8 / np.sqrt((cs0[qq, 0] - cs0[qq, 1])
                 ** 2 + 4 * (1 - cs0[qq, 0] - cs0[qq, 1])))
                index = index + 1
        else:
            for qq in range(nqp):

                pnts[index] = newt_poly(np.array([cs0[qq, 0], cs0[qq, 1]]))

                # evaluate ∂_t at the quadrature points
                p_s = ds_poly(np.array([cs0[qq, 0], cs0[qq, 1]]))

                # evaluate ∂_s at the quadrature points
                p_t = dt_poly(np.array([cs0[qq, 0], cs0[qq, 1]]))

                # Compute ||∂_s x ∂_t||
                J = compute_norm(_cross(p_s, p_t))
                ws[index] = ws0[qq] * J
                index = index + 1
            
    
    return index


def quadrature_split_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                              lp_dgr, fun_handle, deg_integration, quadrature_rule, pnts, ws, index):
    """
    For a mixed mesh, find the cell integration of the test function f.

    Args:
        ls_function: callable
            Zero-levelset function.
        ls_grad_func: callable
            Gradient of the zero-levelset function.
        vertices: ndarray
            Array of vertex coordinates.
        faces: ndarray
            Array of face connectivity.
        interp_deg: int
            Interpolation degree.
        lp_dgr: int
            :math:`l_p`-norm, which is used to define the polynomial degree.
        deg_integration: int
            Degree of integration.
        quadrature_rule: str, optional
            Quadrature rule type ('Gauss_Legendre' or 'Gauss_Simplex').
        pnts: ndarray
            Quadrature points array.
        ws: ndarray
            Quadrature weights array.
        index: int
            Current index in the arrays.

    Returns:
        int
            Updated index value.
    """
    
    for i in range(Refinement):
        vertices, faces = subdivide(vertices, faces)

    index = quadrature_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                                  lp_dgr,fun_handle,deg_integration,quadrature_rule,pnts, ws, index)

    return index
