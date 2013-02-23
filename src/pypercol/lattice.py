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

    def __init__(self, lattice_vectors, frac_coords, decoration=None, 
                 supercell=(1,1,1)):
        """
        Arguments:
          lattice_vectors    3x3 matrix with lattice vectors in rows
          frac_coords        Nx3 array; fractional coordinates of the 
                             N lattice sites
          decoration[i]      initial occupation O of site i (corresponding to
                             frac_coords[i]); 
                             O > 0 --> occupied; O < 0 --> vacant
          supercell          list of multiples of the cell in the three 
                             spacial directions
        """

        """                    static data 

        _avec[i][j]   j-th component of the i-th lattice vector
        _coo[i][j]    j-th component of the coordinates of the i-th 
                      lattice site
        _nsites       total number of lattice sites

        _dNN[i]             nearest neighbor distance from the i-th site
        _nn[i][j]           j-th nearest neighbor site of site i
        _nnn[i][j]          j-th next nearest neighbor site of site i
                            (only computed, if requested)
        _bonds[i][j]        True, if there is a bond between the j-th site 
                            and its j-th neighbor
        _T_nn[i][j]         the translation vector belonging to _nn[i][j]
        _T_nnn[i][j]        the translation vector belonging to _nnn[i][j]
                            (only computed, if requested)
        _nsurface           number of sites at cell boundary 
                            (only computed, if requested)
        """

        self._avec     = (np.array(lattice_vectors).T * supercell).T
        self._coo      = []
        self._occup    = []
        self._occupied = []
        self._vacant   = []

        isite = 0
        for i in range(len(frac_coords)):
            coo = np.array(frac_coords[i])
            if np.any(decoration):
                Oi = decoration[i]
            else:
                Oi = -1
            for ix in xrange(supercell[0]):
                for iy in xrange(supercell[1]):
                    for iz in xrange(supercell[2]):
                        self._coo.append((coo + [ix, iy, iz])/
                                         np.array(supercell, dtype=np.float64))
                        self._occup.append(Oi)
                        if (Oi > 0):
                            self._occupied.append(isite)
                        else:
                            self._vacant.append(isite)
                        isite += 1

        self._coo    = np.array(self._coo)
        self._nsites = len(self._coo)
        self._occup  = np.array(self._occup)

        self._dNN      = []
        self._nn       = []
        self._nnn      = []
        self._T_nn     = []
        self._T_nnn    = []
        self._nsurface = 0

        self._build_neighbor_list()

    @classmethod
    def from_structure(cls, structure, species="O", **kwargs):
        """
        Create a Lattice instance based on the lattice vectors
        defined in a `structure' object (pymatgen.core.structure).

        Arguments:

          structure       an instance of pymatgen.core.structure.Structure
          species         the species that marks occupied lattice sites
                          all other species will be converted to vacant sites

          All keyword arguments of the main constructor are supported, 
          except the initial 'decoration', which is deduced from
          the structure.
        """
    
        avec    = structure.lattice.matrix
        coo     = structure.frac_coords
        symbol  = np.array([s.symbol for s in structure.species])
        decorat = np.where(symbol==species, 1, -1)

        lattice = cls(avec, coo, decoration=decorat, **kwargs)

        return lattice

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        ostr  = "\n Instance of the Lattice class:\n\n"
        ostr += " Lattice vectors:\n\n"
        for v in self._avec:
            ostr += "   {:12.8f}  {:12.8f}  {:12.8f}\n".format(*v)
        ostr += "\n total number of sites: {}".format(self._nsites)
        ostr += "\n occupied sites       : {}".format(self.num_occupied)
        ostr += "\n"
        ostr += str(self._nblist)
        return ostr

    #------------------------------------------------------------------#
    #                            properties                            #
    #------------------------------------------------------------------#

    @property
    def nn(self):
        return list(self._nn)

    @property
    def nnn(self):
        return list(self._nnn)

    @property
    def num_occupied(self):
        return len(self._occupied)

    @property
    def num_vacant(self):
        return len(self._vacant)

    @property
    def num_sites(self):
        return self._nsites

    @property
    def occupied(self):
        return list(self._occupied)

    @property
    def vacant(self):
        return list(self._vacant)

    #------------------------------------------------------------------#
    #                          public methods                          #
    #------------------------------------------------------------------#

    def random_decoration(self, p=0.5, N=None):
        """
        Randomly occupy lattice sites.

        Arguments:
          p    occupation probability
          N    exact number of sites to be occupied

        Note: if specified, N takes precedence over p.
        """

        if not N:
            N = int(np.floor(p*float(self._nsites)))
        N = max(0, min(N, self._nsites))

        idx = np.random.permutation(self._nsites)
        self._occupied = []
        self._vacant   = range(self._nsites)
        for i in range(N):
            self._occupied.append(idx[i])
            del self._vacant[self._vacant.index(idx[i])]
        self._occup[:] = -1
        self._occup[idx[0:N]] = 1


    def get_nnn_shells(self, dr=0.1):
        """
        Calculate shells of next nearest neighbors and store them 
        in `nnn'.
        """
        
        nnn   = []
        T_nnn = []

        pbcdist = self._nblist.get_pbc_distances_and_translations
        for i in xrange(self._nsites):
            nn_i = self._nn[i]
            nnnb = set([])
            for j in nn_i:
                nn_j = self._nn[j]
                nnnb |= set(nn_j) - set(nn_i) - {i}
            nnnb = list(nnnb)
            (dist, Tvecs) = pbcdist(i,nnnb[0])
            dmin = dist[0]
            nnn_i = []
            T_nnn_i = []
            for j in nnnb:
                (dist, Tvec) = pbcdist(i,j)
                for k in xrange(len(dist)):
                    if (dist[k] < dmin - dr):
                        nnn_i   = [j]
                        T_nnn_i = [Tvec[k]]
                        dmin    = dist[k]
                    elif (dist[k] <= dmin + dr):
                        nnn_i.append(j)
                        T_nnn_i.append(Tvec[k])
            nnn.append(nnn_i)
            T_nnn.append(T_nnn_i)

        self._nnn   = nnn
        self._T_nnn = T_nnn


    def save_structure(self, file_name="CONTCAR", vacant="V", occupied="O"):
        """
        Save current occupation to an output file.
        Relies on `pymatgen' for the file I/O.

        Arguments:
          file_name    name of the output file
          vacant       atomic species to be placed at vacant sites
          occupied     atomic species to be placed at occupied sites
        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vaspio      import Poscar

        species = [vacant for i in range(self._nsites)]
        for i in self._occupied:
            species[i] = occupied

        species = np.array(species)
        idx     = np.argsort(species)

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

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

        self._nblist = nblist
        self._dNN    = dNN
        self._nn     = nbs
        self._T_nn   = Tvecs
        

#----------------------------------------------------------------------#
#                              unit test                               #
#----------------------------------------------------------------------#

if (__name__ == "__main__"): #{{{ unit test 

    print("\n FCC 4x4x4 cell (64 sites)")

    avec = np.array([ [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0] ])*5.0

    coo = np.array([[0.0, 0.0, 0.0]])
    lat = Lattice(avec, coo, supercell=(4,4,4))

    print(lat)

    print(" checking number of nearest neighbors (12 for FCC) ... ", end="")
    passed = True
    for nn_i in lat.nn:
        N_nn = len(nn_i)
        if (N_nn != 12):
            print(N_nn)
            print(nn_i)
            passed = False
            break
    if passed:
        print("passed.")
    else:
        print("FAILED!")
    
    print(" checking number of next nearest neighbors (6 for FCC) ... ", end="")
    lat.get_nnn_shells()
    passed = True
    for nnn_i in lat.nnn:
        N_nnn = len(nnn_i)
        if (N_nnn != 6):
            print(N_nnn)
            print(nnn_i)
            passed = False
            break
    if passed:
        print("passed.")
    else:
        print("FAILED!")

    print(" testing random decoration of 16 (of 64) sites ... ", end="")
    lat.random_decoration(p=0.25)
    N1 = np.sum(np.where(lat._occup>0, 1, 0))
    lat.random_decoration(N=16)
    N2 = np.sum(np.where(lat._occup>0, 1, 0))
    if N1 == N2 == 16:
        print("passed.")
    else:
        print("FAILED.")

    print(" exporting structure to file CONTCAR ... ", end="")
    try:
        lat.save_structure('CONTCAR')
        print("passed.")
    except:
        print("FAILED.")

    print("")

