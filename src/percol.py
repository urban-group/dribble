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

    print("Monte-Carlo simulation:\n")
    print("p       P_s          P_infty      chi")

    p_list  = np.arange(p0,p1,dp)
    P_infty = np.zeros(len(p_list))
    P_s     = np.zeros(len(p_list))
    chi     = np.zeros(len(p_list))
    for ip in range(len(p_list)):

        p = p_list[ip]

        for i in range(N):
            percolator.random_decoration(p)
            # percolator.find_all_clusters()
            P_s[ip]     += percolator.find_spanning_cluster()
            P_infty[ip] += 0 #percolator.p_infinity
            chi[ip]     += 0 #percolator.susceptibility

        P_s[ip]     /= float(N)
        P_infty[ip] /= float(N)
        chi[ip]     /= float(N)

        print("%5.4f  %10.8f  %10.8f  %15.8e" % (
            p, P_s[ip], P_infty[ip], chi[ip]))

    # fit with sigmoidal function
    print("\nFitting with sigmoidal function... ", end="")
    (x0, y0, l, fac) = (0.5, 0.01, 10.0, 1.0)
    (popt, pconv)  = curve_fit( stepfunc1, p_list, P_infty, 
                                p0=(x0, y0, l, fac) )
    print("done.  pc = {}\n".format(popt[0]))


    # output to files

    datafile = "percol.out"
    fitfile  = "percol.fit"

    with open(datafile, "w") as f:
        f.write("# p     P_s         P_infty     chi\n")
        for i in range(len(p_list)):
            f.write("{}  {} {}  {}\n".format(p_list[i], P_s[i], P_infty[i], chi[i]))

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

    parser.add_argument(
        "--debug",
        help    = "run in debugging mode",
        action  = "store_true" )

    args = parser.parse_args()

    if args.debug:
        np.random.seed(seed=1)

    percol( poscarfile  = args.structure, 
            percolating = args.percolating )



