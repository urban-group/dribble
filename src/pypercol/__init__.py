"""
Numerical calculaion of the site and bond percolation of an 
arbitrary crystal lattice.
"""

__author__ = "Alexander Urban"
__date__   = "2013-01-15"

class Percolator:
    
    def __init__(self, structure):
        """
        structure     an instance of pymatgen.core.structure.Structure
        """
        self._structure = structure

    def get_site_percolation(self):
        return 0.0

    def get_bond_percolation(self):
        return 0.0

    def my_dummy_func(self):
        return None

