#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-02-20"

import argparse
import sys

from pymatgen.io.vaspio import Poscar
from pypercol           import Lattice
from pypercol           import IsingModel
from pypercol           import Percolator
from pypercol.ising     import k_B
from pypercol.aux       import ProgressBar
from pypercol.aux       import uprint

#----------------------------------------------------------------------#

def runmc(infile, T=300.0, v1=0.5e-3, v2=0.5e-3, H=0.0, Nequi=100, 
          NMC=500, Nstruc=250, supercell=(1,1,1), common=None, 
          conc=None, opt=False):

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
    ising = IsingModel(lattice, v1, v2, H=H)
    uprint(" done.")
    uprint(ising)

    tau    = 1.0
    kT_inv = 1.0/(k_B*T)

    Nevery = int(round(float(NMC)/float(Nstruc)))

    if opt:
        Nequi = NMC
        NMC = 0
        Tfinal = 600.0
        Tramp = (Tfinal - T)/(Nequi-1)
        Nconst_max = 50
        Nconst = 0
        E_prev = 0.0

    with open(outfile, 'w', 0) as f:

        if opt:
            uprint(" annealing for {} MC steps from T = {} to {}".format(Nequi,T,Tfinal))
        else:
            uprint(" equilibrating for {} MC steps at T = {}".format(Nequi,T))

        conv = False
        pb = ProgressBar(Nequi)
        for i in xrange(Nequi):
            pb()
            E = ising.mc_NVT(kT_inv=kT_inv, tau=tau)
            f.write("{} {} {}\n".format(i, E, T))
            if opt and (i < Nequi-1):
                T += Tramp
                kT_inv = 1.0/(k_B*T)
                if (E == E_prev):
                    Nconst += 1
                    if (Nconst >= Nconst_max):
                        uprint(" Converged after {} MC steps.\n".format(i))
                        conv = True
                        break
                else:
                    E_prev = E
                    Nconst = 0
        if not conv:
            pb()

        if (NMC > 0):
            uprint(" now sampling {} structures ".format(Nstruc)
                   + "out of {} MC steps".format(NMC))
            
            nsamp = 0
            nspan = 0.0
            pb = ProgressBar(NMC)
            for i in xrange(Nequi,NMC+Nequi):
                pb()
                E = ising.mc_NVT(kT_inv=kT_inv, tau=tau)
                f.write("{} {}\n".format(i, E))
                if (i % Nevery == 0):
                    nsamp += 1
                    nspan += percol.check_spanning()
            pb()
            
            uprint(" percolation probability: {}".format(
                float(nspan)/float(nsamp)))

        E_tot = ising.total_energy()
        if (abs(E-E_tot)>1.0e-6):
            uprint("Error: final energy inconsistent - check MC code!")
        
        uprint("  E_tot         T             V1            "
               + "V2            H             V1/kT         p")
        uprint(" {:.6e}  {:.6e}  {:.6e}  {:.6e}  {:.6e}  {:.6e}  {:.6f}\n".format(
            E_tot, T, v1, v2, H, v1*kT_inv, 
            float(lattice.num_occupied)/float(lattice.num_sites)))
                
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
        "--V1", 
        help    = "Nearest neighbor interaction.",
        type    = float,
        default = 0.3e-2)

    parser.add_argument(
        "--V2", 
        help    = "Next nearest neighbor interaction.",
        type    = float,
        default = 0.3e-2)

    parser.add_argument(
        "-H", "--magnetfield", 
        help    = "Magnetic field term (point interaction).",
        type    = float,
        default = 0.0,
        dest    = "H")

    parser.add_argument(
        "--opt", 
        help    = "Only optimize structure (search ground state).",
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
        default = 500,
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
           v1        = args.V1,
           v2        = args.V2,
           H         = args.H,
           Nequi     = args.N_equi,
           NMC       = args.N_MC,
           Nstruc    = args.N_samp,
           supercell = args.supercell,
           common    = args.common,
           conc      = args.conc,
           opt       = args.opt)
