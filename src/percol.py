#!/usr/bin/env python

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

def sigmoid(x, x0, y0, l, f):
    return f/(1.0 + np.exp(-l*(x-x0))) + y0

#----------------------------------------------------------------------#

def percol(poscarfile, percolating='Li'):

    input_struc = Poscar.from_file(poscarfile).structure
    percolator  = Percolator(input_struc, percolating)

    N = 1000

    xdata = []
    ydata = []

    for p in np.arange(0.05,0.31,0.01):

        P_infinity = 0.0
        for i in range(N):
            percolator.random_decoration(p)
            percolator.find_all_clusters()
            P_infinity += percolator.p_infinity

        xdata.append(p)
        ydata.append(P_infinity/float(N))

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # fit with sigmoidal function
    (x0, y0, l, f) = (0.5, 0.01, 30.0, 1.0)
    (popt, pconv)  = curve_fit(sigmoid, xdata, ydata, p0=(x0, y0, l, f))

    


#----------------------------------------------------------------------#

if __name__=="__main__":

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



