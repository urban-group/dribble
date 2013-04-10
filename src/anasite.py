#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-04-05"

import argparse
import numpy as np
import sys

from scipy.spatial import Delaunay

from pymatgen.io.vaspio import Poscar
from pypercol.pynblist  import NeighborList

EPS   = 100.0*np.finfo(np.float).eps

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


#---------------------- heights in a tetrahedron ----------------------#

def tetrahedron_heights(nodes):
    """
    Calculates and returns the three four heights of an irregular tetrahedron.

    Arguments:
      nodes == [A, B, C, D]   list of corners of the tetrahedron with
                              P == (p_x, p_y, p_2) being the Cartesian
                              coordinates of point P for P in A, B, C, D

    Returns:
      volume list of heights
    """

    (A, B, C, D) = nodes

    AB = B - A
    AC = C - A
    AD = D - A
    BC = C - B
    BD = D - B
    
    # normal vectors on the tetrahedron faces
    n = np.cross(AB,AC)
    n_ABC = n/np.linalg.norm(n)
    n = np.cross(AC,AD)
    n_ACD = n/np.linalg.norm(n)
    n = np.cross(AB,AD)
    n_ABD = n/np.linalg.norm(n)
    n = np.cross(BC,BD)
    n_BCD = n/np.linalg.norm(n)

    # heights
    h_A = abs(np.dot(AB,n_BCD))
    h_B = abs(np.dot(AB,n_ACD))
    h_C = abs(np.dot(AC,n_ABD))
    h_D = abs(np.dot(AD,n_ABC))

    return np.sort([h_A, h_B, h_C, h_D])

#------------------- circumsphere of a tetrahedron --------------------#

def circumsphere(nodes):

    (A, B, C, D) = nodes
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    f  = A[2]*(B[1]*(D[0]-C[0]) - C[1]*(D[0]-B[0]) + D[1]*(C[0]-B[0])) 
    f -= B[2]*(A[1]*(D[0]-C[0]) - C[1]*(D[0]-A[0]) + D[1]*(C[0]-A[0])) 
    f += C[2]*(A[1]*(D[0]-B[0]) - B[1]*(D[0]-A[0]) + D[1]*(B[0]-A[0])) 
    f -= D[2]*(A[1]*(C[0]-B[0]) - B[1]*(C[0]-A[0]) + C[1]*(B[0]-A[0]))

    if (f <= EPS):
        sys.stderr.write(
            "Error: There is no circumsphere. "
            + "All point lie in the same plain.")
        sys.exit()

    f = 0.5/f

    xyz1 = A*A
    xyz2 = B*B
    xyz3 = C*C
    xyz4 = D*D

    d21  = xyz2 - xyz1
    d31  = xyz3 - xyz1
    d41  = xyz4 - xyz1
    d32  = xyz3 - xyz2
    d42  = xyz4 - xyz2
    d43  = xyz4 - xyz3

    x0   =  f*( A[2]*(B[1]*d43 - C[1]*d42 + D[1]*d32) 
              - B[2]*(A[1]*d43 - C[1]*d41 + D[1]*d31) 
              + C[2]*(A[1]*d42 - B[1]*d41 + D[1]*d21) 
              - D[2]*(A[1]*d32 - B[1]*d31 + C[1]*d21) )

    y0   = -f*( A[2]*(B[0]*d43 - C[0]*d42 + D[0]*d32) 
              - B[2]*(A[0]*d43 - C[0]*d41 + D[0]*d31) 
              + C[2]*(A[0]*d42 - B[0]*d41 + D[0]*d21) 
              - D[2]*(A[0]*d32 - B[0]*d31 + C[0]*d21) )

    z0   =  f*( A[1]*(B[0]*d43 - C[0]*d42 + D[0]*d32) 
              - B[1]*(A[0]*d43 - C[0]*d41 + D[0]*d31) 
              + C[1]*(A[0]*d42 - B[0]*d41 + D[0]*d21) 
              - D[1]*(A[0]*d32 - B[0]*d31 + C[0]*d21) )

    O = np.array([x0,y0,z0])
    R = np.linalg.norm(O - A)
 
    return (O, R)


#----------------------------------------------------------------------#

def analyze_sites(infile, r_cut, site_species='Li', frame_species='O',
                  tet_only=False, verbose=False):

    struc  = Poscar.from_file(infile).structure
    avec   = np.array(struc.lattice.matrix)
    coords = np.array(struc.frac_coords)
    cart   = np.dot(coords, avec)
    types  = np.array([s.symbol for s in struc.species])
    
    nblist = NeighborList(coords, lattice_vectors=avec, types=types, 
                          interaction_range=r_cut)

    sites = np.arange(len(coords))[types == site_species]

    distances = []
    volumes   = []

    for s in sites:

        (nbl, d, T) = nblist.get_neighbors_and_distances(s)
        idx = (types[nbl] == frame_species)
        nbl = np.array(nbl)[idx]
        T   = np.array(T)[idx]
        d   = np.array(d)[idx]

        if (tet_only and len(nbl) != 4):
            continue

        d_min = np.min(d)
        
        # nodes defining the site polyhedron
        nodes = cart[nbl] + np.dot(T, avec)

        # compute volume as sum of the volumina of the 
        # Delaunay tetrahedrons 
        vertices = Delaunay(nodes).vertices
        V  = 0.0
        for tet in vertices:
            V += tetrahedron_volume(nodes[tet])

        if tet_only:
            H = tetrahedron_heights(nodes)
            print("{:5d}  {:2s} {:2s} {:2d} {:8.4f} {:6.2f} {:8.4f}".format(
                s+1, types[s], frame_species, len(nodes), d_min, V, H[0]))            
        else:
            print("{:5d}  {:2s} {:2s} {:2d} {:8.4f} {:6.2f}".format(
                s+1, types[s], frame_species, len(nodes), d_min, V))

        distances.append(d_min)
        volumes.append(V)

    if (len(volumes) > 0) and verbose:

        V_av  = np.sum(volumes)/len(volumes)
        V_max = np.max(volumes)
        V_min = np.min(volumes)
    
        d_min_av  = np.sum(distances)/len(distances)
        d_min_min = np.min(distances)

        print("")
        print(" average site volume: {:.2f} A^3".format(V_av))
        print(" maximum site volume: {:.2f} A^3".format(V_max))
        print(" minimum site volume: {:.2f} A^3".format(V_min))
        print("")
        print(" average minimal {}-{} distance: {:.3f} A".format(
            site_species, frame_species, d_min_av))
        print(" global minimal {}-{} distance : {:.3f} A".format(
            site_species, frame_species, d_min_min))
        print("")

    sys.stdout.flush()

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "input_file",
        help    = "Input file in VASP's POSCAR format.")

    parser.add_argument(
        "--range", "-r",
        help    = "Interaction range (cut-off redius).",
        type    = float,
        default = "3.0",
        dest    = "r_cut")

    parser.add_argument(
        "--site-species",
        help    = "Species whose sites shall be analyzed (default: Li).",
        type    = str,
        default = "Li")

    parser.add_argument(
        "--frame-species",
        help    = "Species that geometrically defines sites (default: O).",
        type    = str,
        default = "O")

    parser.add_argument(
        "--tetrahedrons", "-t",
        help    = "Only analyse tetrahedral sites.",
        action  = "store_true")

    parser.add_argument(
        "--verbose", "-v",
        help    = "Print summary of the results.",
        action  = "store_true")

    args = parser.parse_args()

    analyze_sites( infile        = args.input_file,
                   r_cut         = args.r_cut,
                   site_species  = args.site_species,
                   frame_species = args.frame_species,
                   tet_only      = args.tetrahedrons,
                   verbose       = args.verbose )

