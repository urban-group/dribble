#!/usr/bin/env python

import argparse
import sys
import time

import numpy as np

try:
    from pymatgen.io.vaspio import Poscar
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from pypercol     import Percolator
from pypercol.aux import uprint

#----------------------------------------------------------------------#

def percol(poscarfile, samples, save_clusters=False, 
           file_name="percol.out", pc_site=False, pinf=False, pwrap=False):

    uprint("\n Initializing structure and percolator ... ", end="")

    struc = Poscar.from_file(poscarfile).structure
    percolator = Percolator.from_structure(struc)

    uprint("done.\n")

    uprint(" MC percolation simulation")
    uprint(" -------------------------\n")

    if pc_site:

        #--------------------------------------------------------------#
        #            calculate critical site concentrations            #
        #--------------------------------------------------------------#

        if save_clusters:
            (pc_any, pc_two, pc_all
            ) = percolator.find_percolation_point(
                samples=samples, file_name=file_name+".cluster")
        else:
            (pc_any, pc_two, pc_all
            ) = percolator.find_percolation_point(samples=samples)

        uprint(" Wrapping cluster in any direction at            p = {}".format(pc_any))
        uprint(" Wrapping cluster in at least two directions at  p = {}".format(pc_two))
        uprint(" Wrapping cluster in all three directions at     p = {}".format(pc_all))

        uprint("")

    if pinf:

        #--------------------------------------------------------------#
        #                    estimate P_infinity(p)                    #
        #--------------------------------------------------------------#

        plist = np.arange(0.01, 1.00, 0.01)
        (Q, X) = percolator.calc_p_infinity(plist, samples=samples)

        # integrate susceptibility X in order to normalize it
        intX = np.sum(X)*(plist[1]-plist[0])

        fname = file_name + ".infty"
        uprint(" Writing results to: {}\n".format(fname))

        with open(fname, 'w') as f:
            f.write("# {:^10s}   {:>10s}   {:>15s}   {:>15s}\n".format(
                "p", "P_infty(p)", "Chi(p)", "normalized"))
            for p in xrange(len(plist)):
                f.write("  {:10.8f}   {:10.8f}   {:15.8f}   {:15.8f}\n".format(
                    plist[p], Q[p], X[p], X[p]/intX))

    if pwrap:

        #--------------------------------------------------------------#
        #                    estimate P_wrapping(p)                    #
        #--------------------------------------------------------------#

        plist = np.arange(0.01, 1.00, 0.01)
        (Q, Qc) = percolator.calc_p_wrapping(plist, samples=samples)

        fname = file_name + ".wrap"
        uprint(" Writing results to: {}\n".format(fname))

        with open(fname, 'w') as f:
            f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
                "p", "P_wrap(p)", "cumulative"))
            for p in xrange(len(plist)):
                f.write("  {:10.8f}   {:10.8f}   {:10.8f}\n".format(
                    plist[p], Q[p], Qc[p]))
        

    dt = time.gmtime(time.clock())
    print(" All done.  Elapsed CPU time: {:02d}h{:02d}m{:02d}s\n".format(
        dt.tm_hour, dt.tm_min, dt.tm_sec))

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument(
        "structure",
        help    = "structure in VASP's extended POSCAR format", 
        default = "POSCAR",
        nargs   = "?")

    parser.add_argument(
        "--pc-site", "-p",
        help    = "Calculate critical site concentrations",
        action  = "store_true")

    parser.add_argument(
        "--pinf", "-s",
        help    = "Estimate P_infinity and percolation susceptibility",
        action  = "store_true")

    parser.add_argument(
        "--pwrap", "-w",
        help    = "Estimate P_wrap(p)",
        action  = "store_true")

    parser.add_argument(
        "--samples",
        help    = "number of samples to be averaged",
        type    = int,
        default = 500)

    parser.add_argument(
        "--file-name",
        help    = "base file name for all output files",
        default = "percol")

    parser.add_argument(
        "--save-clusters",
        help    = "save wrapping clusters to file",
        action  = "store_true")

    parser.add_argument(
        "--debug",
        help    = "run in debugging mode",
        action  = "store_true" )

    args = parser.parse_args()

    if args.debug:
        np.random.seed(seed=1)

    percol( poscarfile    = args.structure, 
            samples       = args.samples,
            save_clusters = args.save_clusters,
            file_name     = args.file_name,
            pc_site       = args.pc_site,
            pinf          = args.pinf,
            pwrap         = args.pwrap )



