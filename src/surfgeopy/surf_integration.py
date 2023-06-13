#Imports
import numpy as np
from minterpy import MultiIndexSet, Grid, NewtonPolynomial
from minterpy.dds import dds

# Local imports
from .quadrature_points import *
from .remesh import *
from .utils import *

__all__ = ['integration', 'compute_surf_quadrature', 'quadrature_surf_tri',
           'SimpleImplicitSurfaceProjection']


def integration(ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement,
                fun_handle=lambda _: 1.0, deg_integration=-1):
    """
    Compute integration of a function over curved triangles.
    
    Args:
        ls_function: Zero-levelset function
        ls_grad_func: Gradient of zero-levelset function
        vertices: Array of vertex coordinates
        faces: Array of face connectivity
        interp_deg: Interpolation degree
        lp_dgr: :math:`l_p`-norm, which is used to define the polynomial degree
        Refinement: Refinement level
        fun_handle: Function to be evaluated on each quadrature point (default: lambda _: 1.0)
        deg_integration: Degree of integration (default: -1, use default configuration)
        
    Returns:
        integration value
    """
    
    if (deg_integration > 0):
        # if the degree is set up, use the given integration and interpolation degree
        pnts, ws, offset = compute_surf_quadrature(ls_function, ls_grad_func, vertices, faces,
                                                     interp_deg, lp_dgr, Refinement, 100, deg_integration)

    else:
        # Use the default con_facesiguration if the integration and interpolation degree is not set up
        pnts, ws, offset = compute_surf_quadrature(
            ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement)

    # Perform a function evaluation on each curved triangle
    n_faces = faces.shape[0]
    fs = [0] * n_faces

    for fun_id in range(n_faces):
        for pid in range(offset[fun_id], offset[fun_id + 1]):
            fs[fun_id] += fun_handle(pnts[pid]) * ws[pid]

    return fs


def compute_surf_quadrature(ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement, deg_integration=14):
    """Compute quadrature points and weights on curved triangles.
 
    Args:
        ls_function: Zero-levelset function
        ls_grad_func: Gradient of zero-levelset function
        vertices: Array of vertex coordinates
        faces: Array of face connectivity
        interp_deg: Interpolation degree
        lp_dgr: :math:`l_p`-norm, which is used to define the polynomial degree
        Refinement: Refinement level
        deg_integration: Degree of integration (default: 14)
        
    Returns:
        Quadrature points, weights, and offset array
    """

    # initialization
    index = 0
    n_faces = faces.shape[0]
    nv_surf = faces.shape[1]
    max_nv = max(1000000, n_faces * 6)
    pnts = np.array([[0.0] * 3 for pid in range(max_nv)])
    ws = np.array([0.0 for pid in range(max_nv)])
    offset = np.array([0 for fun_id in range(n_faces + 1)])

    # go through all the faces
    for fun_id in range(n_faces):
        offset[fun_id] = index

        n_elem = nv_surf - 1
        while faces[fun_id, n_elem] < 0:
            n_elem -= 1
        if n_elem < 2:
            continue

        # split each element into several curved triangles
        for j in range(1, n_elem):
            lvids = [0, j, j + 1]
            pnts_tri = vertices[faces[fun_id, lvids]]
            
            # generate quadrature points
            if Refinement > 0:
                index = quadrature_split_surf_tri(pnts_tri, np.array([[0, 1, 2]]), ls_function, ls_grad_func,
                                                    interp_deg, lp_dgr, Refinement, deg_integration, pnts, ws, index)
            else:
                index = quadrature_surf_tri(ls_function, ls_grad_func,
                                              pnts_tri, np.array([[0, 1, 2]]), interp_deg, lp_dgr, deg_integration, pnts, ws, index)

    pnts = pnts[:index]
    ws = ws[:index]
    offset[n_faces] = index
    return pnts, ws, offset


def quadrature_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                          lp_dgr, deg_integration, pnts, ws, index):
    """For a mixed mesh, find the cell integration of the test function f. 

    Args:
        ls_function: Zero-levelset function
        ls_grad_func: Gradient of zero-levelset function
        vertices: Array of vertex coordinates
        faces: Array of face connectivity
        interp_deg: Interpolation degree
        lp_dgr: :math:`l_p`-norm, which is used to define the polynomial degree
        deg_integration: Degree of integration
        pnts: Quadrature points array
        ws: Quadrature weights array
        index: Current index in the arrays

    Returns:
        Updated index value
    """
    n_faces = faces.shape[0]
    n = interp_deg
    lp = lp_dgr
    pnts_q = np.zeros((1, 3), dtype=np.float64)
    pnts_qq = np.zeros((1, 3), dtype=np.float64)
    ws0, cs =quadrule_on_flat(deg_integration)
    nqp = ws0.shape[0]
    mi = MultiIndexSet.from_degree(
        spatial_dimension=2, poly_degree=n, lp_degree=lp)
    grid = Grid(mi)

    # Transform Chebyshev points from [-1,1]^2 to the reference simplex.
    generating_points = transform(grid.unisolvent_nodes)
    quad_ps = np.array([[(1.0 - generating_points[row1, 0] - generating_points[row1, 1]),
                       generating_points[row1, 0], generating_points[row1, 1]] for row1 in range(generating_points.shape[0])])
    w, cs0 = quadrule_on_simplex(deg_integration)

    # Transform quadrature points from the reference simplex to a unit square
    ksi = inv_transform(cs0)
    # enlarge the size of quadrature points buffer if inadequate
    if index + n_faces * nqp > len(ws):
        n_new = 2 * len(ws) + n_faces * nqp
        ws.resize(n_new, refcheck=False)
        pnts.resize((n_new, 3), refcheck=False)
        
    for fun_id in range(n_faces):
        pnts_p = np.array(
            [[0.0] * 3 for pid in range(grid.unisolvent_nodes.shape[0])])
        for q in range(quad_ps.shape[0]):
            pnts_qq = (
                quad_ps[q, 0] * vertices[faces[fun_id, 0]]
                + quad_ps[q, 1] * vertices[faces[fun_id, 1]]
                + quad_ps[q, 2] * vertices[faces[fun_id, 2]])

            pnts_p[q] = SimpleImplicitSurfaceProjection(
                ls_function, ls_grad_func, pnts_qq)

        interpol_coeffs =np.squeeze(dds(pnts_p, grid.tree))
        newt_poly  = NewtonPolynomial(mi, interpol_coeffs)
  
        # compute partial derivatives with respect to "s"
        ds_poly =newt_poly.diff([1, 0])

        # compute partial derivatives with respect to "t"
        dt_poly =newt_poly.diff([0, 1])


        for qq in range(nqp):
            # there are two alternatives to project quadrature points on the  manifold
            # 1. via closest point projection
            #             pnts[index] = SimpleImplicitSurfaceProjection(ls_function, ls_grad_func,pnts_q)
            # 2.  by evaluating the polynomial, which is less exepnsive
            pnts[index] = newt_poly(np.array([ksi[qq, 0], ksi[qq, 1]]))

            # evaluate $\Delta_{s}$ at the quadrature points
            x1s = ds_poly(np.array([ksi[qq, 0], ksi[qq, 1]]))

            # evaluate $\Delta_{t}$ at the quadrature points
            x1t = dt_poly(np.array([ksi[qq, 0], ksi[qq, 1]]))

           # compute $||\Delta_{s}x$\Delta_{t}$||$
            J = compute_norm(_cross(x1s, x1t))
            # Please use this in the case you are applying Duffy' transform
           #ws[index] = ws0[qq] * J * (4/(1-cs0[qq, 1]))
            ws[index] = ws0[qq] * J * \
                (8 / np.sqrt((cs0[qq, 0] - cs0[qq, 1])
                 ** 2 + 4 * (1 - cs0[qq, 0] - cs0[qq, 1])))
            index = index + 1
    
    return index


def quadrature_split_surf_tri(vertices, faces, ls_function, ls_grad_func, interp_deg, lp_dgr,
                                Refinement, deg_integration, pnts, ws, index):
    """ Subdivide a mesh into smaller triangles and compute quadrature points and weights.
    
    Args:
        vertices: Array of vertex coordinates
        faces: Array of face connectivity
        ls_function: Zero-levelset function
        ls_grad_func: Gradient of zero-levelset function
        interp_deg: Interpolation degree
        lp_dgr: math:`l_p`-norm, which is used to define the polynomial degree
        Refinement: Number of mesh refinements
        deg_integration: Degree of integration
        pnts: Quadrature points array
        ws: Quadrature weights array
        index: Current index in the arrays
        
    Returns:
        Updated index value
    """
    for i in range(Refinement):
        vertices, faces = subdivide(vertices, faces)

    index = quadrature_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                                  lp_dgr, deg_integration, pnts, ws, index)

    return index




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
