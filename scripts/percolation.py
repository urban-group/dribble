#!/usr/bin/env python

import argparse
import sys
import time

import numpy as np

from percol.io import Input

try:
    from pymatgen.io.vasp import Poscar
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from percol import Percolator
from percol import Lattice
from percol.aux import uprint

__author__ = "Alexander Urban"


def check_if_percolating(percolator):
    noccup = percolator.num_occupied
    nspan = percolator.check_spanning(verbose=True)
    if (nspan > 0):
        uprint(" The initial structure is percolating.\n")
        uprint(" Fraction of accessible sites: {}\n".format(
            float(nspan)/float(noccup)))
    else:
        uprint(" The initial structure is NOT percolating.\n")
        uprint(" Fraction of accessible sites: 0.0\n")


def calc_critical_concentration(percolator, save_clusters, samples,
                                file_name, sequence):
    if save_clusters:
        (pc_site_any, pc_site_two, pc_site_all, pc_bond_any,
         pc_bond_two, pc_bond_all) = percolator.percolation_point(
             sequence, samples=samples, file_name=file_name+".vasp")
    else:
        (pc_site_any, pc_site_two, pc_site_all, pc_bond_any,
         pc_bond_two, pc_bond_all) = percolator.percolation_point(
             sequence, samples=samples)

    uprint(" Critical site (bond) concentrations to find a "
           "wrapping cluster\n")

    uprint(" in one or more dimensions   p_c1 = {:.8f}  ({:.8f})".format(
        pc_site_any, pc_bond_any))
    uprint(" in two or three dimensions  p_c2 = {:.8f}  ({:.8f})".format(
        pc_site_two, pc_bond_two))
    uprint(" in all three dimensions     p_c3 = {:.8f}  ({:.8f})".format(
        pc_site_all, pc_bond_all))

    uprint("")


def calc_p_infinity(percolator, samples, save_raw, file_name, sequence):
    plist = np.arange(0.01, 1.00, 0.01)
    (Q, X) = percolator.calc_p_infinity(
        plist, sequence, samples=samples,
        save_discrete=save_raw)

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


def calc_p_wrapping(percolator, samples, save_raw, file_name, sequence):
    plist = np.arange(0.01, 1.00, 0.01)
    (Q, Qc) = percolator.calc_p_wrapping(
        plist, sequence, samples=samples,
        save_discrete=save_raw)

    fname = file_name + ".wrap"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
            "p", "P_wrap(p)", "cumulative"))
        for p in xrange(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:10.8f}\n".format(
                plist[p], Q[p], Qc[p]))


def calc_inaccessible_sites(percolator, samples, save_raw, file_name,
                            sequence, species):
    plist = np.arange(0.01, 1.00, 0.01)
    (F_inacc, nclus) = percolator.inaccessible_sites(
        plist, sequence, species, samples=samples,
        save_discrete=save_raw)

    fname = file_name + ".inacc"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
            "p", "F_inacc(p)", "N_percol(p)"))
        for p in xrange(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:12.6f}\n".format(
                plist[p], F_inacc[p], nclus[p]))


def compute_percolation(input_file, samples, save_clusters, save_raw,
                        file_name, pc, check, pinf, pwrap, inaccessible,
                        supercell):

    if not (check or pc or pinf or pwrap or inaccessible):
        print("\n Nothing to do.")
        print(" Please specify the quantity to be calculated.")
        print(" Use the `--help' flag to list all options.\n")
        sys.exit()

    uprint("\n Parsing input file '{}'...".format(input_file), end="")
    inp = Input.from_file(input_file)
    uprint(" done.")

    uprint("\n Setting up lattice and neighbor lists...", end="")
    lattice = Lattice.from_structure(
        inp.mg_structure, supercell=supercell, NN_range=inp.cutoff,
        site_labels=inp.site_labels,
        occupying_species=inp.percolating_species,
        static_species=inp.static_species,
        concentrations=inp.initial_occupancy,
        formula_units=inp.formula_units)
    uprint(" done.")
    uprint(" Initial site occupations taken from structure file.")
    uprint(lattice)

    uprint(" Initializing percolator...", end="")
    percolator = Percolator(lattice,
                            percolating_species=inp.percolating_species,
                            static_species=inp.static_species,
                            initial_concentrations=inp.initial_occupancy)
    uprint(" done.")

    percolator.set_complex_percolation_rule(
        site_rules=inp.sublattices, bond_rules=inp.bonds, verbose=True)

    uprint("\n MC percolation simulation\n -------------------------\n")

    # always use initial occaptions
    if check:  # check, if initial structure is percolating
        check_if_percolating(percolator)
    if pc:  # calculate critical site concentrations
        calc_critical_concentration(percolator, save_clusters, samples,
                                    file_name, inp.flip_sequence)
    if pinf:  # estimate P_infinity(p)
        calc_p_infinity(percolator, samples, save_raw, file_name,
                        inp.flip_sequence)
    if pwrap:  # estimate P_wrapping(p)
        calc_p_wrapping(percolator, samples, save_raw, file_name,
                        inp.flip_sequence)
    if inaccessible is not None:  # fraction of inaccessible sites
        calc_inaccessible_sites(percolator, samples, save_raw,
                                file_name, inp.flip_sequence,
                                inaccessible)

    dt = time.gmtime(time.clock())
    uprint(" All done.  Elapsed CPU time: {:02d}h{:02d}m{:02d}s\n".format(
            dt.tm_hour, dt.tm_min, dt.tm_sec))


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "input_file",
        help="Input file in JSON format")

    parser.add_argument(
        "--supercell",
        help="List of multiples of the lattice cell" +
             " in the three lattice directions",
        type=int,
        default=(1, 1, 1),
        nargs="+")

    parser.add_argument(
        "--inaccessible", "-i",
        help="Calculate fraction of inaccessible sites for given "
             "reference species",
        type=str,
        default=None,
        metavar="SPECIES")

    parser.add_argument(
        "--pc", "-p",
        help="Calculate critical site concentrations",
        action="store_true")

    parser.add_argument(
        "--check",
        help="Check, if the initial structure is percolating.",
        action="store_true")

    parser.add_argument(
        "--pinf", "-s",
        help="Estimate P_infinity and percolation susceptibility",
        action="store_true")

    parser.add_argument(
        "--pwrap", "-w",
        help="Estimate P_wrap(p)",
        action="store_true")

    parser.add_argument(
        "--samples",
        help="number of samples to be averaged",
        type=int,
        default=500)

    parser.add_argument(
        "--file-name",
        help="base file name for all output files",
        default="percol")

    parser.add_argument(
        "--save-clusters",
        help="save wrapping clusters to file",
        action="store_true")

    parser.add_argument(
        "--save-raw",
        help="Also store raw data before convolution (where available).",
        action="store_true")

    parser.add_argument(
        "--debug",
        help="run in debugging mode",
        action="store_true")

    args = parser.parse_args()

    if args.debug:
        np.random.seed(seed=1)

    compute_percolation(input_file=args.input_file,
                        samples=args.samples,
                        save_clusters=args.save_clusters,
                        save_raw=args.save_raw,
                        file_name=args.file_name,
                        pc=args.pc,
                        check=args.check,
                        pinf=args.pinf,
                        pwrap=args.pwrap,
                        inaccessible=args.inaccessible,
                        supercell=args.supercell)
