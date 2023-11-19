surfgeopy Documentation
=======================

Welcome to the documentation for surfgeopy, a Python package for calculating surface integrals over smooth embedded manifolds.

.. toctree::
   :maxdepth: 2
   :hidden:

   install
   modules
   examples

Introduction
------------

``surfgeopy`` is an open-source Python package designed for approximating surface integrals over smooth embedded manifolds. It employs curved surface triangulations through k-th order interpolation of the closest point projection. This extends initial linear surface approximations.

Square-Squeezing Technique
--------------------------

The key innovation in ``surfgeopy`` lies in the parametrization of triangles :math:`T_i` over squares using a square-squeezing technique. This transformative approach reconfigures interpolation tasks of triangulated manifolds to the standard hypercube through a recently introduced cube-to-simplex transformation. This innovative process enhances the accuracy of surface approximations, making ``surfgeopy`` a powerful tool for high-fidelity calculations.


Chebyshev-Lobatto Grids
-----------------------

To guarantee stability and accuracy in computations, ``surfgeopy`` leverages classic Chebyshev-Lobatto grids. These grids enable the calculation of high-order interpolants for surface geometry while avoiding Runge's phenomenon, a common issue in numerical analysis.

Surface Approximation Using Polynomial Interpolation
-----------------------------------------------------

Consider an element :math:`T_i` in a reference surface :math:`S_h`. We consider the affine transformation and closest point projection:

- :math:`\tau_i : \Delta_2 \rightarrow T_i`
- :math:`\pi_i : T_i \rightarrow S`

Setting

- :math:`\varphi_i : \Omega \rightarrow S, \quad \varphi_i = \pi_i \circ \tau_i \circ \sigma`
  where :math:`\sigma` is a mapping from the reference triangle :math:`\Delta_2` to the physical triangle :math:`T_i`.

Let

- :math:`q_i(\mathrm{x}) = \sum_{\alpha \in A_{2,n}} \varphi_i(p_\alpha)L_{\alpha}(\mathrm{x})`
  be an :math:`n`-th order polynomial (Newton) interpolation of the mapping :math:`\varphi_i` on :math:`\Omega`.

Then,

- :math:`T_n = \{q_i, \Omega, q_i(\Omega)\}_{i=1,\dots,N}`
  is an :math:`n`-th order approximation of the smooth surface :math:`S`.

.. figure:: images/approximation_frame.jpg
   :alt: Surface Approximation
   :width: 4000


Square-Triangle Transformation
===============================

Square-triangle transformations: Deformations of an equidistant grid (left picture) under Duffy's transformation (middle picture) and square-squeezing (right picture)

.. image:: images/square_equi.png
   :alt: drawing
   :width: 200


.. image:: images/dufyy_equi.png
   :alt: Duffy's Transformation
   :width: 200


.. image:: images/ss_equi.png
   :alt: Square-Squeezing
   :width: 200

   

