#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-02-20"

import argparse

from pymatgen.io.vaspio import Poscar
from pypercol           import Lattice
from pypercol           import IsingModel

#----------------------------------------------------------------------#

def runmc(infile, supercell=(1,1,1)):

    structure = Poscar.from_file(infile).structure
    lattice = Lattice.from_structure(structure, supercell=supercell)
    
    T       = 300
    v1      =  1.0
    v2      = -0.3
    mcsteps = 100
    
    ising = IsingModel(lattice, v1=v1, v2=v2)
    
    for i in xrange(mcsteps):
        E = ising.mc(T)
        print("{} {}".format(i, E))

    lattice.save_structure('CONTCAR')

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
        "--supercell",
        help    = "List of multiples of the lattice cell" +
                  " in the three spacial directions",
        type    = int,
        default = (1,1,1),
        nargs   = "+")

    args = parser.parse_args()

    runmc( infile    = args.structure,
           supercell = args.supercell )
