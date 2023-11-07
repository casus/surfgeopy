#Imports
import numpy as np
from minterpy import MultiIndexSet, Grid, NewtonPolynomial
import minterpy as mp
from minterpy.dds import dds
from numba import njit

# Local imports
from .quadrature_points import quadrule_on_flat,quadrule_on_simplex
from .remesh import subdivide
from .utils import (SimpleImplicitSurfaceProjection,compute_norm, 
                       read_mesh_data,transform,inv_transform,_cross)

__all__ = ['integration', 'compute_surf_quadrature', 'quadrature_surf_tri',
           'quadrature_split_surf_tri']


def integration(ls_function, ls_grad_func, mesh, interp_deg, lp_dgr, Refinement,
                fun_handle=lambda _: 1.0, deg_integration=-1):
    """
        Compute integration of a function over curved triangles.
        
        Args:
            ls_function: callable
                Zero-levelset function.
            ls_grad_func: callable
                Gradient of zero-levelset function.
            mesh (str): 
                 The file path to the MAT file containing mesh data.
            vertices: ndarray
                Array of vertex coordinates.
            faces: ndarray
                Array of face connectivity.
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
            
        Returns:
                Integration values for each curved triangle.

    """
    
    vertices, faces=read_mesh_data(mesh)
    if (deg_integration > 0):
        # if the degree is set up, use the given integration and interpolation degree
        pnts, ws, offset = compute_surf_quadrature(ls_function, ls_grad_func, vertices, faces,
                                                     interp_deg, lp_dgr, Refinement,fun_handle, deg_integration)

    else:
        # Use the default configuration if the integration and interpolation degree is not set up
        pnts, ws, offset = compute_surf_quadrature(
            ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement,fun_handle)

    # Perform a function evaluation on each curved triangle
    n_faces = faces.shape[0]
    fs = [0] * n_faces

    for fun_id in range(n_faces):
        for pid in range(offset[fun_id], offset[fun_id + 1]):
            fs[fun_id] += fun_handle(pnts[pid]) * ws[pid]

    return fs


def compute_surf_quadrature(ls_function, ls_grad_func, vertices, faces, interp_deg, lp_dgr, Refinement,fun_handle, deg_integration=14):
    """"
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

        Returns:
            ndarray, ndarray, ndarray
                Quadrature points, weights, and offset array.
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
                                                    interp_deg, lp_dgr, Refinement,fun_handle, deg_integration, pnts, ws, index)
            else:
                index = quadrature_surf_tri(ls_function, ls_grad_func,
                                              pnts_tri, np.array([[0, 1, 2]]), interp_deg, lp_dgr,fun_handle, deg_integration, pnts, ws, index)

    pnts = pnts[:index]
    ws = ws[:index]
    offset[n_faces] = index
    return pnts, ws, offset


def quadrature_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                          lp_dgr,fun_handle, deg_integration, pnts, ws, index):
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
    ws0, cs =quadrule_on_simplex(interp_deg)
    nqp = ws0.shape[0]
    mi = MultiIndexSet.from_degree(
        spatial_dimension=2, poly_degree=n, lp_degree=lp)
    grid = Grid(mi)

    # Transform Chebyshev points from [-1,1]^2 to the reference simplex.
    generating_points = transform(grid.unisolvent_nodes,duffy_transform=False)
    quad_ps = np.array([[(1.0 - generating_points[row1, 0] - generating_points[row1, 1]),
                       generating_points[row1, 0], generating_points[row1, 1]] for row1 in range(generating_points.shape[0])])
    w, cs0 = quadrule_on_simplex(interp_deg)

    # Transform quadrature points from the reference simplex to a unit square
    ksi = inv_transform(cs0,duffy_transform=False)
    # enlarge the size of quadrature points buffer if inadequate
    if index + n_faces * nqp > len(ws):
        n_new = 2 * len(ws) + n_faces * nqp
        ws.resize(n_new, refcheck=False)
        pnts.resize((n_new, 3), refcheck=False)
    for fun_id in range(n_faces):
        pnts_p = np.array(
            [[0.0] * 3 for pid in range(grid.unisolvent_nodes.shape[0])])
        pnts_func = np.array(
            [[0.0] * 1 for pid in range(grid.unisolvent_nodes.shape[0])])
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
                                Refinement,fun_handle, deg_integration, pnts, ws, index):
    """
        Subdivide a mesh into smaller triangles and compute quadrature points and weights.
        
        Args:
            vertices: ndarray
                Array of vertex coordinates.
            faces: ndarray
                Array of face connectivity.
            ls_function: callable
                Zero-levelset function.
            ls_grad_func: callable
                Gradient of zero-levelset function.
            interp_deg: int
                Interpolation degree.
            lp_dgr: int
                :math:`l_p`-norm, which is used to define the polynomial degree.
            Refinement: int
                Number of mesh refinements.
            deg_integration: int
                Degree of integration.
            pnts: ndarray
                Quadrature points array.
            ws: ndarray
                Quadrature weights array.
        
    Returns:
        Updated index
    """
    for i in range(Refinement):
        vertices, faces = subdivide(vertices, faces)

    index = quadrature_surf_tri(ls_function, ls_grad_func, vertices, faces, interp_deg,
                                  lp_dgr,fun_handle, deg_integration, pnts, ws, index)

    return index