#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

import numpy as np
from scipy.optimize import curve_fit

try:
    import pymatgen
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from pymatgen.io.vaspio       import Poscar
from pypercol                 import Percolator

#----------------------------------------------------------------------#

def stepfunc1(x, x0, y0, a, b):
    return a*np.tanh(b*(x-x0)) + y0

#----------------------------------------------------------------------#

def percol(poscarfile, percolating='Li'):

    print("\nInitializing the percolator... ", end="")

    input_struc = Poscar.from_file(poscarfile).structure
    percolator  = Percolator(input_struc, percolating)

    print("done.\n")

    N = 5000
    p0 = 0.05
    dp = 0.01
    p1 = 0.31

    print("Calculating P_infinity:\n")

    data = {}
    for p in np.arange(p0,p1,dp):

        print("p = {} ... ".format(p), end="")

        P_infinity = 0.0
        for i in range(N):
            percolator.random_decoration(p)
            percolator.find_all_clusters()
            P_infinity += percolator.p_infinity

        data[p] = P_infinity/float(N)

        print("done.  P_infinity = {}".format(data[p]))

    data = np.array([data.keys(), data.values()]).transpose()
    data = np.sort(data, axis=0)

    # fit with sigmoidal function
    print("\nFitting with sigmoidal function... ", end="")
    (x0, y0, l, fac) = (0.5, 0.01, 10.0, 1.0)
    (popt, pconv)  = curve_fit( stepfunc1, data[:,0], data[:,1], 
                                p0=(x0, y0, l, fac) )
    print("done.  pc = {}\n".format(popt[0]))


    # output to files

    datafile = "percol.out"
    fitfile  = "percol.fit"

    with open(datafile, "w") as f:
        f.write("# p     P_infinity\n")
        for i in range(len(data)):
            f.write("{}  {}\n".format(data[i,0], data[i,1]))

    with open(fitfile, "w") as f:
        f.write("# f(x) = a*tanh(b*(x-x0)) + y0\n")
        f.write("# x0 = {}; y0 = {}; a = {}; b = {}\n".format(*popt))
        f.write("# pc = x0 = {}\n".format(popt[0]))
        for p in np.arange(p0,p1,0.1*dp):
            f.write("{}  {}\n".format(p, stepfunc1(p, *popt)))
        

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

    args = parser.parse_args()

    percol( poscarfile  = args.structure, 
            percolating = args.percolating )



