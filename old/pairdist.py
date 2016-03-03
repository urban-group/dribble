#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-04-08"

import argparse

from pymatgen.io.vaspio import Poscar


def pairdistance(infile, i, j):

    struc  = Poscar.from_file(infile).structure

    s1 = struc.sites[i-1]
    s2 = struc.sites[j-1]

    print(s1.distance(s2))

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "input_file",
        help    = "Input file in VASP's POSCAR format.")

    parser.add_argument(
        "site1",
        help    = "Number of first site.",
        type = int)

    parser.add_argument(
        "site2",
        help    = "Number of second site.",
        type = int)

    args = parser.parse_args()

    pairdistance( args.input_file, args.site1, args.site2 )

