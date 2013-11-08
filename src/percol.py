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
from pypercol     import Lattice
from pypercol.aux import uprint

#----------------------------------------------------------------------#

def percol(poscarfile, samples, save_clusters=False, save_raw=False,
           file_name="percol.out", pc=False, check=False, r_NN=None,
           pinf=False, pwrap=False, bonds=False, flux=False,
           inaccessible=False, supercell=[1,1,1], common=0, same=None,
           require_NN=False, occupations=False):

    if not (check or pc or pinf or pwrap or bonds or flux or inaccessible):
        print("\n Nothing to do.")
        print(" Please specify the quantity to be calculated.")
        print(" Use the `--help' flag to list all options.\n")
        sys.exit()

    uprint("\n Reading structure from file '{}'...".format(poscarfile), end="")
    struc = Poscar.from_file(poscarfile).structure
    uprint(" done.")

    uprint("\n Setting up lattice and neighbor lists...", end="")
    lattice = Lattice.from_structure(struc, supercell=supercell, NN_range=r_NN)
    uprint(" done.")
    uprint(" Initial site occupations taken from structure file.")
    if occupations:
        uprint(" These occupations will be used, but in random order.")
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

    uprint("\n MC percolation simulation")
    uprint(" -------------------------\n")

    if check:

        #--------------------------------------------------------------#
        #          check, if initial structure is percolating          #
        #--------------------------------------------------------------#

        noccup = percolator.num_occupied
        nspan  = percolator.check_spanning(verbose=True)

        if (nspan > 0):
            uprint(" The initial structure is percolating.\n")
            uprint(" Fraction of accessible sites: {}\n".format(
                float(nspan)/float(noccup)))
        else:
            uprint(" The initial structure is NOT percolating.\n")
            uprint(" Fraction of accessible sites: 0.0\n")

    if pc:

        #--------------------------------------------------------------#
        #            calculate critical site concentrations            #
        #--------------------------------------------------------------#

        if save_clusters:
            (pc_site_any, pc_site_two, pc_site_all,
             pc_bond_any, pc_bond_two, pc_bond_all,
            ) = percolator.percolation_point(
                samples=samples,
                file_name=file_name+".vasp",
                initial_occupations=occupations)
        else:
            (pc_site_any, pc_site_two, pc_site_all,
             pc_bond_any, pc_bond_two, pc_bond_all,
           ) = percolator.percolation_point(
                samples=samples,
                initial_occupations=occupations)

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
        (Q, X) = percolator.calc_p_infinity(
            plist, samples=samples,
            save_discrete=save_raw,
            initial_occupations=occupations)

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
        (Q, Qc) = percolator.calc_p_wrapping(
            plist, samples=samples,
            save_discrete=save_raw,
            initial_occupations=occupations)

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
        F_bonds = percolator.percolating_bonds(
            plist, samples=samples,
            save_discrete=save_raw,
            initial_occupations=occupations)

        fname = file_name + ".bonds"
        uprint(" Writing results to: {}\n".format(fname))

        with open(fname, 'w') as f:
            f.write("# {:^10s}   {:>10s}\n".format(
                "p", "F_bonds(p)"))
            for p in xrange(len(plist)):
                f.write("  {:10.8f}   {:10.8f}\n".format(
                    plist[p], F_bonds[p]))

    if flux:

        #--------------------------------------------------------------#
        #                       percolation flux                       #
        #--------------------------------------------------------------#

        plist = np.arange(0.01, 1.00, 0.01)
        flux = percolator.percolation_flux(
            plist, samples=samples,
            save_discrete=save_raw,
            initial_occupations=occupations)

        fname = file_name + ".flux"
        uprint(" Writing results to: {}\n".format(fname))

        with open(fname, 'w') as f:
            f.write("# {:^10s}   {:>10s}\n".format(
                "p", "F_bonds(p)"))
            for p in xrange(len(plist)):
                f.write("  {:10.8f}   {:10.8f}\n".format(
                    plist[p], flux[p]))

    if inaccessible:

        #--------------------------------------------------------------#
        #                fraction of inaccessible sites                #
        #--------------------------------------------------------------#

        plist = np.arange(0.01, 1.00, 0.01)
        (F_inacc, nclus) = percolator.inaccessible_sites(
            plist, samples=samples,
            save_discrete=save_raw,
            initial_occupations=occupations)

        fname = file_name + ".inacc"
        uprint(" Writing results to: {}\n".format(fname))

        with open(fname, 'w') as f:
            f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
                "p", "F_inacc(p)", "N_percol(p)"))
            for p in xrange(len(plist)):
                f.write("  {:10.8f}   {:10.8f}   {:12.6f}\n".format(
                    plist[p], F_inacc[p], nclus[p]))

    dt = time.gmtime(time.clock())
    uprint(" All done.  Elapsed CPU time: {:02d}h{:02d}m{:02d}s\n".format(
            dt.tm_hour, dt.tm_min, dt.tm_sec))

#----------------------------------------------------------------------#
#                        command line arguments                        #
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
        "--bonds", "-b",
        help    = "Calculate fraction of percolating bonds",
        action  = "store_true")

    parser.add_argument(
        "--flux", "-f",
        help    = "Calculate percolation flux",
        action  = "store_true")

    parser.add_argument(
        "--inaccessible", "-i",
        help    = "Calculate fraction of inaccessible sites",
        action  = "store_true")

    parser.add_argument(
        "--pc", "-p",
        help    = "Calculate critical site concentrations",
        action  = "store_true")

    parser.add_argument(
        "--check",
        help    = "Check, if the initial structure is percolating.",
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

    parser.add_argument(
        "--use-occupations",
        help    = "Use the (randomized) occupations of the initial structure.",
        action  = "store_true")

    parser.add_argument(
        "--file-name",
        help    = "base file name for all output files",
        default = "percol")

    parser.add_argument(
        "--save-clusters",
        help    = "save wrapping clusters to file",
        action  = "store_true")

    parser.add_argument(
        "--save-raw",
        help    = "Also store raw data before convolution (where available).",
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
            save_raw      = args.save_raw,
            file_name     = args.file_name,
            pc            = args.pc,
            check         = args.check,
            r_NN          = args.NN_range,
            pinf          = args.pinf,
            pwrap         = args.pwrap,
            bonds         = args.bonds,
            flux          = args.flux,
            inaccessible  = args.inaccessible,
            supercell     = args.supercell,
            common        = args.common,
            same          = args.same,
            require_NN    = args.require_NN,
            occupations   = args.use_occupations)
