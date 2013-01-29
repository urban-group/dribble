#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

import numpy as np

try:
    from pymatgen.io.vaspio import Poscar
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from pypercol import Percolator

#----------------------------------------------------------------------#

def percol(poscarfile, percolating='Li'):

    print("\nInitializing structure and percolator ... ", end="")

    struc = Poscar.from_file(poscarfile).structure
    percolator = Percolator.from_structure(struc)

    print("done.\n")

    print("MC percolation simulation\n")

    (pc, pc_all) = percolator.find_percolation_point(
        samples=500, file_name="./clusters/POSCAR")
    print("pc1 = {}".format(pc[0]))
    print("pc2 = {}".format(pc[1]))
    print("pc3 = {}".format(pc[2]))
    print("pc  = {}".format(pc_all))
    print("")


#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument(
        "structure",
        help    = "structure in VASP's extended POSCAR format", 
        default = "POSCAR",
        nargs   = "?")

    parser.add_argument(
        "--percolating", "-p",
        help    = "the percolating species",
        dest    = "percolating",
        default = "Li" )

    parser.add_argument(
        "--debug",
        help    = "run in debugging mode",
        action  = "store_true" )

    args = parser.parse_args()

    if args.debug:
        np.random.seed(seed=1)

    percol( poscarfile  = args.structure, 
            percolating = args.percolating )



