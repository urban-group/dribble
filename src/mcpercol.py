#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-02-20"

import argparse
import sys
import numpy as np

from pymatgen.io.vaspio import Poscar
from pypercol           import Lattice
from pypercol           import IsingModel
from pypercol           import Percolator
from pypercol.ising     import k_B
from pypercol.aux       import ProgressBar
from pypercol.aux       import uprint

EPS   = 100.0*np.finfo(np.float).eps

#----------------------------------------------------------------------#

def runmc(infile, T=300.0, J1=0.5e-3, J2=0.5e-3, H=0.0, Nequi=100, 
          NMC=500, Nstruc=250, supercell=(1,1,1), common=None, 
          conc=None, nocc=None, opt=False, lgh=False, mVT=False):

    outfile = "mcpercol.out"

    uprint("\n Reading structure from file '{}'...".format(infile), end="")
    structure = Poscar.from_file(infile).structure
    uprint(" done.")

    uprint("\n Setting up lattice and neighbor lists...", end="")
    lattice = Lattice.from_structure(structure, supercell=supercell)
    lattice.get_nnn_shells()
    uprint(" done.")
    if conc:
        uprint(" Random site occupations. p = {}".format(conc))
        lattice.random_decoration(p=conc)
    elif nocc:
        uprint(" Random site occupations. N = {}".format(nocc))
        lattice.random_decoration(N=nocc)       
    else:
        uprint(" Initial site occupations taken from structure file.")
    uprint(lattice)

    if not opt:
        uprint(" Initializing percolator...", end="")
        percol = Percolator(lattice)
        uprint(" done.")
        if common > 0:
            uprint(" Using percolation rule with {} common neighbor(s).".format(common))
            percol.set_special_percolation_rule(num_common=common)

    uprint("\n Initializing Ising model...", end="")
    if lgh:
        ising = IsingModel.from_lattice_gas_H(lattice, J1, J2, mu=H)
    else:
        ising = IsingModel(lattice, J1, J2, H=H)
    uprint(" done.")
    uprint(ising)

    tau    = 1.0
    kT_inv = 1.0/(k_B*T)

    Nevery = int(round(float(NMC)/float(Nstruc)))

    if opt:
        Nequi = NMC
        NMC = 0
        Tfinal = 1.0
        Tramp = (Tfinal - T)/(Nequi-1)
        Nconst_max = 30
        Nconst = 0
        E_prev = 0.0
        MC_eps = EPS*float(Nequi)

    with open(outfile, 'w', 0) as f:

        if opt:
            uprint(" annealing for {} MC steps from T = {} to {}".format(Nequi,T,Tfinal))
            E_low = ising.total_energy()
        else:
            uprint(" equilibrating for {} MC steps at T = {}".format(Nequi,T))

        if mVT:
            uprint(" simulating a micro-canonical ensemble (muVT)\n")
        else:
            uprint(" simulating a canonical ensemble (NVT)\n")

        f.write("# step energy temperature concentration\n")

        conv = False
        pb = ProgressBar(Nequi)
        for i in xrange(Nequi):
            pb()
            if mVT:
                E = ising.mc_mVT(kT_inv=kT_inv, tau=tau)
            else:
                E = ising.mc_NVT(kT_inv=kT_inv, tau=tau)
            p = float(lattice.num_occupied)/float(lattice.num_sites)
            f.write("{} {} {} {}\n".format(i, E, T, p))
            if opt and (i < Nequi-1):
                T += Tramp
                T = max(T,1.0)
                kT_inv = 1.0/(k_B*T)
                E_low = min(E, E_low)
                if (abs(E - E_prev) <= MC_eps):
                    Nconst += 1
                    if (Nconst >= Nconst_max) and (E == E_low):
                        uprint(" done.")
                        uprint(" Converged after {} MC steps (T = {}).\n".format(i, T))
                        conv = True
                        break
                else:
                    E_prev = E
                    Nconst = 0
        if not conv:
            pb()
            if opt and (abs(E - E_low) > MC_eps):
                uprint(" Warning: final state is {} higher in energy ".format(E-E_low)
                       + "then lowest encountered !")

        p_percol = 0.0
        f_percol = 0.0
        if (NMC > 0):
            uprint(" now sampling {} structures ".format(Nstruc)
                   + "out of {} MC steps".format(NMC))
            
            nsamp = 0    # number of sampled structures
            nwrap = 0.0  # sum of fractions of percolating sites
            nperc = 0    # number of percolating structures
            pb = ProgressBar(NMC)
            for i in xrange(Nequi,NMC+Nequi):
                pb()
                if mVT:
                    E = ising.mc_mVT(kT_inv=kT_inv, tau=tau)
                else:
                    E = ising.mc_NVT(kT_inv=kT_inv, tau=tau)
                p = float(lattice.num_occupied)/float(lattice.num_sites)
                f.write("{} {} {} {}\n".format(i, E, T, p))
                if (i % Nevery == 0):
                    nsamp += 1
                    wrapping_fraction = percol.check_spanning()
                    if (wrapping_fraction > 0.0):
                        nwrap += wrapping_fraction
                        nperc += 1
            pb()

            p_percol = float(nperc)/float(nsamp)
            f_percol = float(nwrap)/float(nsamp)
            
            uprint(" percolation probability (fraction): {} ({})".format(
                p_percol, f_percol))

        E_tot = ising.total_energy()
        if (abs(E-E_tot)>1.0e-6):
            uprint("Error: final energy inconsistent - check MC code!")
        
        uprint("  E_tot         T             J1            "
               + "J2            H             J1/kT         p"
               + "              p_percol      f_percol")
        uprint(" {:.6e}  {:.6e}  {:.6e}  {:.6e}  ".format(E_tot, T, J1, J2)
               + "{:.6e}  {:.6e}  {:.6f}  {:.6f}  {:.6f}\n".format(
                 H, J1*kT_inv, float(lattice.num_occupied)/float(lattice.num_sites),
                 p_percol, f_percol))
                
    uprint(" Saving final structure to file 'CONTCAR'.")
    lattice.save_structure('CONTCAR')

    uprint("")

#----------------------------------------------------------------------#

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        description     = __doc__+"\n{} {}".format(__date__,__author__),
        formatter_class = argparse.RawDescriptionHelpFormatter )

    parser.add_argument(
        "structure",
        help    = "structure in VASP's extended POSCAR format", 
        default = "POSCAR",
        nargs   = "?")

    parser.add_argument(
        "--common",
        help    = "Number of common neighbors for two sites to be percolating.",
        type    = int,
        default = 0)

    parser.add_argument(
        "-c", "--concentration",
        help    = "Concentration of occupied sites.",
        type    = float,
        default = None,
        dest    = "conc")

    parser.add_argument(
        "-n", "--num-occupied",
        help    = "Exact number of occupied sites.",
        type    = int,
        default = None,
        dest    = "nocc")

    parser.add_argument(
        "--supercell",
        help    = "List of multiples of the lattice cell" +
                  " in the three spacial directions",
        type    = int,
        default = (1,1,1),
        nargs   = "+")

    parser.add_argument(
        "-T", "--temperature",
        help    = "MC temperature.",
        type    = float,
        default = 300.0,
        dest    = "T")

    parser.add_argument(
        "--V1", "--J1",
        help    = "Nearest neighbor interaction.",
        type    = float,
        dest    = "J1",
        default = 0.3e-2)

    parser.add_argument(
        "--V2", "--J2",
        help    = "Next nearest neighbor interaction.",
        type    = float,
        dest    = "J2",
        default = 0.3e-2)

    parser.add_argument(
        "-H", "--mu", 
        help    = "Magnetic field term (point interaction).",
        type    = float,
        default = 0.0,
        dest    = "H")

    parser.add_argument(
        "--opt", 
        help    = "Only optimize structure (search ground state).",
        action  = "store_true")

    parser.add_argument(
        "--LGH", 
        help    = "Initialize as Lattice gas Hamiltonian (v1, v2, mu).",
        action  = "store_true")

    parser.add_argument(
        "--mVT", 
        help    = "Micro-canonical ensemble (variable particle number).",
        action  = "store_true")

    parser.add_argument(
        "--N-equi",
        help    = "Number of MC steps for equilibration.",
        type    = int,
        default = 100,
        dest    = "N_equi")

    parser.add_argument(
        "--N-MC",
        help    = "Number of MC steps for the sampling (after equilibration).",
        type    = int,
        default = 250,
        dest    = "N_MC")

    parser.add_argument(
        "--N-samp",
        help    = "Number of structures to sample for percolation properties.",
        type    = int,
        default = 250,
        dest    = "N_samp")

    args = parser.parse_args()

    runmc( infile    = args.structure,
           T         = args.T,
           J1        = args.J1,
           J2        = args.J2,
           H         = args.H,
           Nequi     = args.N_equi,
           NMC       = args.N_MC,
           Nstruc    = args.N_samp,
           supercell = args.supercell,
           common    = args.common,
           conc      = args.conc,
           nocc      = args.nocc,
           opt       = args.opt,
           lgh       = args.LGH,
           mVT       = args.mVT)
