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
           file_name="percol.out", pc=False, pinf=False, pwrap=False, 
           bonds=False, supercell=[1,1,1], common=0):

    if not (pc or pinf or pwrap or bonds):
        print("\n Nothing to do.")
        print(" Please specify the quantity to be calculated.")
        print(" Use the `--help' flag to list all options.\n")
        sys.exit()

    uprint("\n Initializing structure and percolator ... ", end="")

    struc = Poscar.from_file(poscarfile).structure
    percolator = Percolator.from_structure(struc, supercell=supercell)

    if common > 0:
        percolator.set_special_percolation_rule(num_common=common)

    uprint("done.\n")

    uprint(" MC percolation simulation")
    uprint(" -------------------------\n")

    if pc:

        #--------------------------------------------------------------#
        #            calculate critical site concentrations            #
        #--------------------------------------------------------------#

        if save_clusters:
            (pc_site_any, pc_site_two, pc_site_all,
             pc_bond_any, pc_bond_two, pc_bond_all,
            ) = percolator.find_percolation_point(
                samples=samples, file_name=file_name+".cluster")
        else: 
            (pc_site_any, pc_site_two, pc_site_all,
             pc_bond_any, pc_bond_two, pc_bond_all,
           ) = percolator.find_percolation_point(samples=samples)

        uprint(" Critical site (bond) concentrations to find a wrapping cluster\n")

        uprint(" in one or more dimensions   p_c = {:.8f}  ({:.8f})".format(
            pc_site_any, pc_bond_any))
        uprint(" in two or three dimensions  p_c = {:.8f}  ({:.8f})".format(
            pc_site_two, pc_bond_two))
        uprint(" in all three dimensions     p_c = {:.8f}  ({:.8f})".format(
            pc_site_all, pc_bond_all))

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
        
    if bonds:

        #--------------------------------------------------------------#
        #                fraction of percolating bonds                 #
        #--------------------------------------------------------------#

        plist = np.arange(0.01, 1.00, 0.01)
        F_bonds = percolator.percolating_bonds(plist, samples=samples)

        fname = file_name + ".bonds"
        uprint(" Writing results to: {}\n".format(fname))

        with open(fname, 'w') as f:
            f.write("# {:^10s}   {:>10s}\n".format(
                "p", "F_bonds(p)"))
            for p in xrange(len(plist)):
                f.write("  {:10.8f}   {:10.8f}\n".format(
                    plist[p], F_bonds[p]))

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
        "--supercell",
        help    = "List of multiples of the lattice cell" +
                  " in the three spacial directions",
        type    = int,
        default = (1,1,1),
        nargs   = "+")

    parser.add_argument(
        "--bonds", "-b",
        help    = "Calculate fraction of percolating bonds",
        action  = "store_true")

    parser.add_argument(
        "--pc", "-p",
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
        "--common",
        help    = "Number of common neighbors for two sites to be percolating.",
        type    = int,
        default = 0)

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
            pc            = args.pc,
            pinf          = args.pinf,
            pwrap         = args.pwrap,
            bonds         = args.bonds,
            supercell     = args.supercell,
            common        = args.common )



