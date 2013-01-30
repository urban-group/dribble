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

def percol(poscarfile, samples):

    print("\n Initializing structure and percolator ... ", end="")

    struc = Poscar.from_file(poscarfile).structure
    percolator = Percolator.from_structure(struc)

    print("done.\n")

    print(" MC percolation simulation\n")

    (pc_any, pc_one, pc_two, pc_all
    ) = percolator.find_percolation_point(samples=samples)

    print(" Percolating in any direction at  p_c = {}".format(pc_any))
    print(" Percolating in one direction at  p_c = {}".format(pc_one))
    print(" Percolating in two directions at p_c = {}".format(pc_two))
    print(" Percolating in all directions at p_c = {}".format(pc_all))

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
        "--samples",
        help    = "number of samples to be averaged",
        type    = int,
        default = 500)

    parser.add_argument(
        "--debug",
        help    = "run in debugging mode",
        action  = "store_true" )

    args = parser.parse_args()

    if args.debug:
        np.random.seed(seed=1)

    percol( poscarfile  = args.structure, 
            samples     = args.samples )



