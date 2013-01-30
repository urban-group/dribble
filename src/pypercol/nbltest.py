#!/usr/bin/env python

"""
Benchmark of the neighbor list.
"""

__author__ = "Alexander Urban"
__date__   = "2013-01-28"

import argparse
import numpy as np
from pynblist import NeighborList
from pymatgen.io.vaspio import Poscar

#----------------------------------------------------------------------#

def nbltest(infile):

    structure = Poscar.from_file(infile).structure
    nbl = NeighborList.from_structure(structure)

    rand = np.random.random_integers(0,nbl.num_coords-1, 50)

    for i in rand:
        # (nl1, dist1, T1) = nbl.get_neighbors_and_distances(i)
        (nl1, dist1, T1) = nbl.get_nearest_neighbors(i)
        (nl2, dist2, T2) = nbl.get_neighbors_and_distances_OLD(i)
        for j in range(len(nl1)):
            k = np.where(nl2==nl1[j])[0][0]
            diff = np.sum(np.abs(T1[j] - T2[k]))
            if diff > 0:
                print T1[j], T2[k]
            diff = abs(dist1[j] - dist2[k])
            if diff > 0.0000001:
                print dist1[j], dist2[k]
    

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "structure",
        help    = "structure in VASP's extended POSCAR format",
        default = "POSCAR",
        nargs   = "?" )

    args = parser.parse_args()

    nbltest(args.structure)
