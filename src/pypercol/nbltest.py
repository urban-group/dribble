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

    for i in np.random.random_integers(0,nbl.num_coords-1, 10):
        nbs = nbl.get_neighbors_and_distances(i)
        print i
        print len(nbs)
        print nbs
        
    

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
