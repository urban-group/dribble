#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-11-07"

import argparse
import sys

import numpy as np

try:
    from pymatgen.io.vaspio import Poscar
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from pypercol     import Percolator
from pypercol     import Lattice
from pypercol.aux import uprint

#----------------------------------------------------------------------#

def replace_sites(poscarfile, file_name="replace.out", supercell=[1,1,1],
                  r_NN=None, common=0, same=None, require_NN=False,
                  nsamples=500):

    uprint("\n Reading structure from file '{}'...".format(poscarfile), end="")
    struc = Poscar.from_file(poscarfile).structure
    uprint(" done.")

    uprint("\n Setting up lattice and neighbor lists...", end="")
    lattice = Lattice.from_structure(struc, supercell=supercell, NN_range=r_NN)
    uprint(" done.")
    uprint(" Initial site occupations taken from structure file.")
    print(lattice)

    uprint(" Initializing percolator...", end="")
    percolator = Percolator(lattice)
    uprint(" done.")

    if (common > 0) or same:
        uprint(" Using percolation rule with {} common neighbor(s).".format(common))
        if require_NN:
            uprint(" Require the common neighbors to be themselves nearest neighbors.")
        uprint(" Require same coordinate: {}".format(same))
        percolator.set_special_percolation_rule(
            num_common=common, same=same, require_NN=require_NN)


    plist = np.arange(0.01, 1.00, 0.01)
    (Q, Qc) = percolator.calc_p_wrapping(plist, samples=nsamples,
                                         initial_occupations=True)

    fname = file_name + ".wrap"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
            "p", "P_wrap(p)", "cumulative"))
        for p in xrange(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:10.8f}\n".format(
                plist[p], Q[p], Qc[p]))

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "structure",
        help    = "Structure in VASP's POSCAR format.")

    parser.add_argument(
        "--file-name",
        help    = "base file name for all output files",
        default = "replace")

    parser.add_argument(
        "--supercell",
        help    = "List of multiples of the lattice cell" +
                  " in the three lattice directions",
        type    = int,
        default = (1,1,1),
        nargs   = "+")

    parser.add_argument(
        "--NN-range",
        help    = "longest expected distance in 1st NN shell.",
        type    = float,
        default = None)

    parser.add_argument(
        "--common",
        help    = "Number of common neighbors for two sites to be percolating.",
        type    = int,
        default = 0)

    parser.add_argument(
        "--same",
        help    = "Require bonding sites to have the same coordinate in direction 0, 1, or 2.",
        type    = int,
        default = None)

    parser.add_argument(
        "--require-NN",
        help    = "Require the comon NNs (defined by using the --common flag) "
                + "to be themselves nearest neighbors.",
        action  = "store_true",
        dest    = "require_NN")

    args = parser.parse_args()

    replace_sites(args.structure,
                  file_name  = args.file_name,
                  supercell  = args.supercell,
                  r_NN       = args.NN_range,
                  common     = args.common,
                  same       = args.same,
                  require_NN = args.require_NN)
