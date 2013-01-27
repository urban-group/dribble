#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

import numpy as np
from scipy.optimize import curve_fit

try:
    from pymatgen.io.vaspio import Poscar
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from pypercol           import Percolator
from pypercol.aux       import sigmoidal1
from pypercol.aux       import binom_conv

#----------------------------------------------------------------------#

def percol(poscarfile, percolating='Li'):

    print("\nInitializing the percolator... ", end="")

    input_struc = Poscar.from_file(poscarfile).structure
    percolator  = Percolator(input_struc, percolating)

    print("done.\n")

    N = percolator.num_sites
    Nav = 10

    print("Monte-Carlo simulation:\n")
    print("n       P_s")
    
    P_s = np.zeros(len(range(N+1)))
    n_one = 0
    n_one_limit = int(float(N)*0.1)
    for n in range(N+1):
        for i in range(Nav):
            percolator.random_decoration_n(n)
            P_s[n] += percolator.find_spanning_cluster()
        P_s[n] /= float(Nav)
        print("%5.4f  %10.8f" % (n, P_s[n]))
        if P_s[n] == 1.0:
            n_one += 1
        else:
            n_one = 0
        if n_one > n_one_limit:
            print("Seems converged. Stopping.")
            P_s[n+1:N] = 1.0
            break
        
    datafile = "percol.out"
    with open(datafile, "w") as f:
        f.write("# p     P_s\n")
        for p in np.arange(0.00, 1.01, 0.01):
              Pp = binom_conv(P_s, range(N+1), N, p)
              f.write("{}  {}\n".format(p, Pp))

#$    print("Monte-Carlo simulation:\n")
#$    print("p       P_s          P_infty      chi")
#$
#$    p_list  = np.arange(p0,p1,dp)
#$    P_infty = np.zeros(len(p_list))
#$    P_s     = np.zeros(len(p_list))
#$    chi     = np.zeros(len(p_list))
#$    for ip in range(len(p_list)):
#$
#$        p = p_list[ip]
#$
#$        for i in range(N):
#$            percolator.random_decoration_p(p)
#$            # percolator.find_all_clusters()
#$            P_s[ip]     += percolator.find_spanning_cluster()
#$            P_infty[ip] += 0 #percolator.p_infinity
#$            chi[ip]     += 0 #percolator.susceptibility
#$
#$        P_s[ip]     /= float(N)
#$        P_infty[ip] /= float(N)
#$        chi[ip]     /= float(N)
#$
#$        print("%5.4f  %10.8f  %10.8f  %15.8e" % (
#$            p, P_s[ip], P_infty[ip], chi[ip]))
#$
#$    # fit with sigmoidal function
#$    print("\nFitting with sigmoidal function... ", end="")
#$    (x0, y0, l, fac) = (0.5, 0.01, 10.0, 1.0)
#$    (popt, pconv)  = curve_fit( sigmoidal1, p_list, P_infty, 
#$                                p0=(x0, y0, l, fac) )
#$    print("done.  pc = {}\n".format(popt[0]))
#$
#$
#$    # output to files
#$
#$    datafile = "percol.out"
#$    fitfile  = "percol.fit"
#$
#$    with open(datafile, "w") as f:
#$        f.write("# p     P_s         P_infty     chi\n")
#$        for i in range(len(p_list)):
#$            f.write("{}  {} {}  {}\n".format(p_list[i], P_s[i], P_infty[i], chi[i]))
#$
#$    with open(fitfile, "w") as f:
#$        f.write("# f(x) = a*tanh(b*(x-x0)) + y0\n")
#$        f.write("# x0 = {}; y0 = {}; a = {}; b = {}\n".format(*popt))
#$        f.write("# pc = x0 = {}\n".format(popt[0]))
#$        for p in np.arange(p0,p1,0.1*dp):
#$            f.write("{}  {}\n".format(p, sigmoidal1(p, *popt)))
        

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



