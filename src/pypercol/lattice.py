#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-02-15"

import numpy as np

from pynblist    import NeighborList

#----------------------------------------------------------------------#

class Lattice(object):

    def __init__(self, lattice_vectors, frac_coords, supercell=(1,1,1)):
        """
        Arguments:
          lattice_vectors    3x3 matrix with lattice vectors in rows
          frac_coords        Nx3 array; fractional coordinates of the 
                             N lattice sites
          supercell          list of multiples of the cell in the three 
                             spacial directions
        """

        """                    static data 

        _avec[i][j]   j-th component of the i-th lattice vector
        _coo[i][j]    j-th component of the coordinates of the i-th 
                      lattice site
        _nsites       total number of lattice sites

        _dNN[i]             nearest neighbor distance from the i-th site
        _neighbors[i][j]    j-th neighbor site of site i
        _bonds[i][j]        True, if there is a bond between the j-th site 
                            and its j-th neighbor
        _nsurface           number of sites at cell boundary 
                            (computed, if needed)
        _T_vectors[i][j]    the translation vector belonging to 
                            _neighbors[i][j]
        _nbonds_tot         maximum number of possible bonds between the sites
        """

        self._avec   = (np.array(lattice_vectors).T * supercell).T
        self._coo    = []
        for i in range(len(frac_coords)):
            coo = np.array(frac_coords[i])
            for ix in xrange(supercell[0]):
                for iy in xrange(supercell[1]):
                    for iz in xrange(supercell[2]):
                        self._coo.append((coo + [ix, iy, iz])/
                                         np.array(supercell, dtype=np.float64))
        self._coo = np.array(self._coo)

        self._nsites    = len(self._coo)
        self._occup     =  np.empty(self._nsites, dtype=int)
        self._dNN       = []
        self._neighbors = []
        self._nsurface  = 0
        self._T_vectors = []

        self._build_neighbor_list()

    @classmethod
    def from_structure(cls, structure, **kwargs):
        """
        Create a Percolator instance based on the lattice vectors
        defined in a `structure' object.

        Arguments:
          structure       an instance of pymatgen.core.structure.Structure
          all keyword arguments of the main constructor
        """
    
        avec   = structure.lattice.matrix
        coo    = structure.frac_coords
        percol = cls(avec, coo, **kwargs)

        return percol

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str  = "\n Instance of the Lattice class\n\n"
        str += " Lattice vectors:\n\n"
        for v in self._avec:
            str += "   {:10.8f}  {:10.8f}  {:10.8f}\n".format(*v)
        str += "\n number of sites: {}".format(self._nsites)
        str += "\n"
        return str

    #------------------------------------------------------------------#
    #                            properties                            #
    #------------------------------------------------------------------#

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def num_vacant(self):
        return len(self._vacant)

    @property
    def num_occupied(self):
        return len(self._occupied)

    @property
    def num_sites(self):
        return self._nsites




    #------------------------------------------------------------------#
    #                         private methods                          #
    #------------------------------------------------------------------#

    def _build_neighbor_list(self, dr=0.1):
        """
        Determine the list of neighboring sites for 
        each site of the lattice.  Allow the next neighbor
        distance to vary about `dr'.
        """

        dNN    = np.empty(self._nsites)
        nbs    = range(self._nsites)
        Tvecs  = range(self._nsites)

        nblist = NeighborList(self._avec, self._coo)
        
        for i in xrange(self._nsites):
            (nbl, dist, T) = nblist.get_nearest_neighbors(i, dr=dr)
            nbs[i]   = nbl
            Tvecs[i] = T
            dNN[i]   = np.min(dist)

        self._dNN       = dNN
        self._neighbors = nbs
        self._T_vectors = Tvecs
        
