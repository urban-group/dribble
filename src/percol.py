#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys

try:
    from pymatgen.io.vaspio import Poscar
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

import pypercol



#----------------------------------------------------------------------#

def percol(poscarfile):

    poscar = Poscar.from_file(poscarfile)
    
    

#----------------------------------------------------------------------#

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument(
        "structure",
        help    = "Structure in VASP's extended POSCAR format.", 
        default = "POSCAR",
        nargs   = "?")

    args = parser.parse_args()

    percol(poscarfile = args.structure)
