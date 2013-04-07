#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-04-05"

import argparse
import numpy as np

from pymatgen.io.vaspio import Poscar
from pypercol.lattice   import Lattice

#----------------- volume of an irregular tetrahedron -----------------#

def tetrahedron_volume(nodes):
    """
    Calculates and returns the volume of an irregular tetrahedron.

    Arguments:
      nodes == [A, B, C, D]   list of corners of the tetrahedron with
                              P == (p_x, p_y, p_2) being the Cartesian
                              coordinates of point P for P in A, B, C, D

    Returns:
      volume V (float)
    """

    (A, B, C, D) = nodes

    # vectors defining the tetrahedron
    AB = B - A
    AC = C - A
    AD = D - A
    
    # normal vector on the ABC plane
    n = np.cross(AB,AC)
    n_len = np.linalg.norm(n)
    n /= n_len

    # area of the triangle ABC
    A_ABC = 0.5*n_len

    # distance of D from the ABC plane
    dist_D = abs(np.dot(AD,n))

    # volume
    V = 1./3.*A_ABC*dist_D

    return V

def tetrahedron_volume_heron(nodes):
    """
    Calculates and returns the volume of an irregular tetrahedron.
    The volume is calculated according to the generalized Heron formula.

    Arguments:
      nodes == [A, B, C, D]   list of corners of the tetrahedron with
                              P == (p_x, p_y, p_2) being the Cartesian
                              coordinates of point P for P in A, B, C, D

    Returns:
      volume V (float)
    """

    (A, B, C, D) = nodes

    # edges of the tetrahedron following the naming convention
    # at http://en.wikipedia.org/wiki/Heron%27s_formula (2013-04-05)
    U = np.linalg.norm(B-A)
    V = np.linalg.norm(C-A)
    W = np.linalg.norm(C-B)
    u = np.linalg.norm(D-C)
    v = np.linalg.norm(D-B)
    w = np.linalg.norm(D-A)

    # auxiliary variables
    X = (w - U + v)*(U + v + w)
    x = (U - v + w)*(v - w + U)
    Y = (u - V + w)*(V + w + u)
    y = (V - w + u)*(w - u + V)
    Z = (v - W + u)*(W + u + v)
    z = (W - u + v)*(u - v + W)
    a = np.sqrt(x*Y*Z)
    b = np.sqrt(X*y*Z)
    c = np.sqrt(X*Y*z)
    d = np.sqrt(x*y*z)
    
    V  = (-a+b+c+d)*(a-b+c+d)*(a+b-c+d)*(a+b+c-d)
    V  = np.sqrt(V)
    V /= (192.0*u*v*w)

    return V


#----------------------------------------------------------------------#

def analyze_sites(infile):

    struc = Poscar.from_file(infile).structure
    lattice = Lattice.from_structure(struc)

    for i in xrange(lattice.num_sites):
        nb = lattice.nn[i]
        for j in nb:
            if (j<i):
                continue
            
    

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "input_file",
        help    = "Input file in VASP's POSCAR format.")

    args = parser.parse_args()

    analyze_sites(infile = args.input_file)
