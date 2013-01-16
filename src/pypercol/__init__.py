"""
Direct calculaion of the site and bond percolation of a crystal lattice.
"""

__author__ = "Alexander Urban"
__email__  = "alexurba@mit.edu"
__date__   = "2013-01-15"

import numpy as np

class Lattice:
    
    def __init__(avec=np.identity(3), sites=[], species=[]):
        """
        avec    lattice vector matrix (NumPy array)
        sites   lattice coordinates of all sites (NumPy array)
        species list of lists of species for each site
                (every site can be occupied by)
        """
