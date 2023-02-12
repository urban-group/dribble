#!/usr/bin/env python3
# ----------------------------------------------------------------------
# This file is part of the 'Dribble' package for percolation simulations.
# Copyright (c) 2013-2018 Alexander Urban (aurban@atomistic.net)
# ----------------------------------------------------------------------
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Mozilla Public License, v. 2.0, for more details.

"""
Dribble - Percolation Simulation on Lattices

Analyze the ionic percolation properties of an input structure.

"""

import argparse
import sys
import time

import numpy as np

from dribble.io import Input
from dribble.percolator import Percolator
from dribble.lattice import Lattice
from dribble.misc import uprint

__author__ = "Alexander Urban"


def check_if_percolating(percolator, inp, save_clusters, tortuosity):
    noccup = percolator.num_occupied
    nspan = percolator.check_spanning(verbose=True,
                                      save_clusters=save_clusters,
                                      static_sites=inp.static_sites)
    if (nspan > 0):
        uprint(" The initial structure is percolating.\n")
        uprint(" Fraction of accessible sites: {}\n".format(
            float(nspan)/float(noccup)))
        if tortuosity:
            for c in percolator.percolating_clusters:
                t_min, t_mean, t_std = percolator.get_tortuosity(c)
                uprint(" Tortuosity of cluster {} (min, mean): ".format(c)
                       + "{:5.3f}, {:5.3f} +/- {:5.3f}".format(
                           t_min, t_mean, t_std))
            uprint("")
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
        for p in range(len(plist)):
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
        for p in range(len(plist)):
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
        for p in range(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:12.6f}\n".format(
                plist[p], F_inacc[p], nclus[p]))


def calc_mean_tortuosity(percolator, samples, file_name, sequence):
    F_tort = percolator.mean_tortuosity(
        sequence, samples=samples)

    fname = file_name + ".tortuosity"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}  {:^10s}   {:s}\n".format(
            "N", "p", "Tortuosity(p)"))
        N = len(F_tort)
        for i, T in enumerate(F_tort):
            f.write("  {:10d}  {:10.8f}   {:10.8f}\n".format(
                i+1, (i+1)/float(N), T))


def compute_percolation(input_file, structure_file, samples,
                        save_clusters, save_raw, file_name, pc, check,
                        pinf, pwrap, inaccessible, tortuosity,
                        mean_tortuosity, supercell):

    if not (check or pc or pinf or pwrap or inaccessible or mean_tortuosity):
        print("\n Nothing to do.")
        print(" Please specify the quantity to be calculated.")
        print(" Use the `--help' flag to list all options.\n")
        sys.exit()

    input_params = {}
    if structure_file is not None:
        uprint("\n Reading structure from file: {}".format(structure_file))
        input_params['structure'] = structure_file

    uprint("\n Parsing input file '{}'...".format(input_file), end="")
    inp = Input.from_file(input_file, **input_params)
    uprint(" done.")

    uprint("\n Setting up lattice and neighbor lists...", end="")
    lattice = Lattice.from_input_object(inp, supercell=supercell)
    uprint(" done.")
    uprint(lattice)

    uprint(" Initializing percolator...", end="")
    percolator = Percolator.from_input_object(inp, lattice, verbose=True)
    uprint(" done.")

    uprint("\n MC percolation simulation\n -------------------------\n")

    if check:  # check, if initial structure is percolating
        check_if_percolating(percolator, inp, save_clusters, tortuosity)
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
    if mean_tortuosity:  # tortuosity as function of concentration
        calc_mean_tortuosity(percolator, samples, file_name,
                             inp.flip_sequence)

    dt = time.gmtime(time.process_time())
    uprint(" All done.  Elapsed CPU time: {:02d}h{:02d}m{:02d}s\n".format(
            dt.tm_hour, dt.tm_min, dt.tm_sec))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "input_file",
        help="Input file in JSON format")

    parser.add_argument(
        "structure_file",
        help="Optional structure file in VASP's POSCAR format.",
        default=None,
        nargs="?")

    parser.add_argument(
        "--supercell",
        help="List of multiples of the lattice cell" +
             " in the three lattice directions",
        type=int,
        default=(1, 1, 1),
        nargs=3)

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
        "--tortuosity", "-t",
        help="Compute tortuosity of the percolating clusters as function "
             "of the concentration.  Together with '--check', only compute "
             "tortuosity of the input structure.",
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

    if args.tortuosity and not args.check:
        mean_tortuosity = True
    else:
        mean_tortuosity = False

    compute_percolation(input_file=args.input_file,
                        structure_file=args.structure_file,
                        samples=args.samples,
                        save_clusters=args.save_clusters,
                        save_raw=args.save_raw,
                        file_name=args.file_name,
                        pc=args.pc,
                        check=args.check,
                        pinf=args.pinf,
                        pwrap=args.pwrap,
                        inaccessible=args.inaccessible,
                        tortuosity=args.tortuosity,
                        mean_tortuosity=mean_tortuosity,
                        supercell=args.supercell)


if (__name__ == "__main__"):
    main()
