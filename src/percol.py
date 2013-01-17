#!/usr/bin/env python

import argparse
import sys

try:
    import pymatgen
except ImportError:
    print("Unable to load the `pymatgen' module.")
    sys.exit()

from pymatgen.io.vaspio       import Poscar
from pymatgen.transformations.standard_transformations \
                              import SubstitutionTransformation

from pypercol.transformations import RandomOrderingTransformation
from pypercol                 import Percolator


#----------------------------------------------------------------------#

def percol(poscarfile, el1, el2, p1, navg):

    subst = {el1 : {el1 : p1, el2 : (1.0-p1)}}

    input_struc      = Poscar.from_file(poscarfile).structure
    trans            = SubstitutionTransformation(subst)
    disordered_struc = trans.apply_transformation(input_struc)

    trans2           = RandomOrderingTransformation()

    p_site = 0
    p_bond = 0

    for i in range(navg):
        struc = trans2.apply_transformation(disordered_struc)
        percolator = Percolator(struc)
        p_site += percolator.get_site_percolation()
        p_bond += percolator.get_bond_percolation()
    
    p_site /= navg
    p_bond /= navg

    return (p_site, p_bond)

#----------------------------------------------------------------------#

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument(
        "structure",
        help    = "structure in VASP's extended POSCAR format", 
        default = "POSCAR",
        nargs   = "?")

    parser.add_argument(
        "--element1",
        help    = "the percolating element",
        dest    = "element1",
        default = "Li" )

    parser.add_argument(
        "--element2",
        help    = "the non-percolating element",
        dest    = "element2",
        default = "Co" )

    parser.add_argument(
        "-p", "--probability",
        help    = "probability of percolating site/species",
        dest    = "probability",
        type    = float,
        default = 0.5 )

    parser.add_argument(
        "-n", "--n-average",
        help    = "number of random structures to average",
        dest    = "naverage",
        type    = int,
        default = 10 )

    args = parser.parse_args()

    (p_site, p_bond) = percol( 
        poscarfile = args.structure, 
        el1        = args.element1,
        el2        = args.element2,
        p1         = args.probability,
        navg       = args.naverage )

    print p_site, p_bond

