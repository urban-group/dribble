"""
Numerical calculaion of the site and bond percolation of an 
arbitrary crystal lattice.
"""

__author__ = "Alexander Urban"
__date__   = "2013-01-15"

import numpy as np
from sys import stderr

import sys

from pynblist import NeighborList

PERCOLATING    = 1
NONPERCOLATING = 0

class Percolator(object):

    def __init__(self, structure, percolating='Li', use_decoration=False):
        """
        structure       an instance of pymatgen.core.structure.Structure
        percolating     the percolating atomic species
        use_decoration  use input structure to determine lattice decoration
        """

        self._structure      = structure
        self._dNN            = 0.0
        self._neighbors      = []
        self._clusters       = []
        self._n_cl_max       = 0
        self._p_infinity     = 0.0
        self._susceptibility = 0.0
        self._n_percolating  = 0

        self._build_neighbor_list()

        self._get_boundaries()

        self._decoration = np.zeros(self.num_sites, dtype=int)
        if use_decoration:
            self.decorate(structure, percolating)

    def __str__(self):
        return str(self._structure.lattice)

    #-------------------------- properties ----------------------------#

    @property
    def clusters(self):
        """
        A list of all clusters of percolating sites found in the 
        structure.  The list is set up by `find_all_clusters()'.
        """
        return self._clusters

    @property
    def decoration(self):
        """
        A one dimensional ndarray with entries for each lattice site.
        Percolating sites have the value PERCOLAING; other sites
        are NONPERCOLATING.
        """
        return self._decoration

    @property
    def dNN(self):
        """
        A list of next neighbor distances for each lattice site.
        The list is constructed by `find_all_clusters()'.
        """
        return self._dNN

    @property
    def neighbors(self):
        """
        A list of lists with next neighbor sites for each lattice site.
        """
        return self._neighbors

    @property
    def num_sites(self):
        """
        The total number of lattice sites.
        See also: n_percolating, n_nonpercolating
        """
        return self._structure.num_sites

    @property
    def n_percolating(self):
        """
        The number of percolating lattice sites.
        See also: num_sites, n_nonpercolating
        """
        return self._n_percolating

    @property
    def n_nonpercolating(self):
        """
        The number of non-percolating lattice sites.
        See also: num_sites, n_percolating
        """
        return self.num_sites - self.n_percolating

    @property
    def p_infinity(self):
        """
        An estimate for the probability of finding an infinite
        percolating cluster in the system.
        The actual value is computed by `find_all_clusters()'.
        See also: p_percolating
        """
        return self._p_infinity

    @property
    def p_percolating(self):
        """
        The actual probability of a lattice site being percolating.
        Note: this value may differ from the target probability
              specified for `random_decoration(p)'
        """
        return float(self.n_percolating)/float(self.num_sites)

    @property
    def structure(self):
        """
        The structure object that the percolator instance is based on.
        """
        return self._structure

    @property
    def susceptibility(self):
        """
        The percolation susceptibility.
        Its actual value is computed by `find_all_clusters()' 
        and by `calc_susceptibility()'.
        """
        return self._susceptibility


    #------------------------------------------------------------------#
    #                             methods                              #
    #------------------------------------------------------------------#

    
    #---------------------- lattice decorations -----------------------#

    def decorate(self, structure, percolating='Li'):
        """
        Use the lattice decoration of STRUCTURE 
        (from pymatgen.core.structure) to set the occupancies of the 
        internal lattice representation.  The percolating species 
        is PERCOLATING.

        STRUCTURE must not be disordered.  Instead it is expected to 
        have definite (integer) site occupancies.

        The only distinction made is between the PERCOLATING species
        and any other non-percolating species.  The actual atom types
        are not stored. This is mainly important to understand how
        randomize_decoration() works.

        """

        if not structure.is_ordered:
            stderr.write("Warning: structure NOT ordered."
                         +" No decoration obtained.")
            return

        if not (structure.num_sites == self.num_sites):
            raise IncompatibleStructureException("Error: incompatible structure.")

        self._n_percolating = 0
            
        for i in range(self.num_sites):
            if (structure.sites[i].specie.symbol == percolating):
                self._decoration[i] = PERCOLATING
                self._n_percolating += 1
            else:
                self._decoration[i] = NONPERCOLATING

    def randomize_decoration(self):
        """
        Randomize the current lattice decoration by simply shuffling the 
        decoration array.
        """

        np.random.shuffle(self._decoration)

    def random_decoration(self, p):
        """
        Randomly decorate lattice with a probability P of the percolating
        species.  0.0 < P < 1.0
        """

        self._decoration[:] = NONPERCOLATING
        r = np.random.random(self.num_sites)
        idx = (r <= p)
        self._decoration[idx] = PERCOLATING
        self._n_percolating = np.sum(np.where(idx, 1, 0))

    #--------------------- percolation analysis -----------------------#

    def get_cluster(self, site, visited=[], pbc=True):
        """
        Recursively determine all percolating sites connected to SITE.
        """

        if (self._decoration[site] == NONPERCOLATING):
            return visited
        
        visited.append(site)
        for (nb,jimg) in self._neighbors[site]:
            if not (pbc or np.all(jimg == 0)):
                continue
            if ((not (nb in visited)) and (self._decoration[nb] == PERCOLATING)):
                visited += self.get_cluster(nb, visited, pbc)[len(visited):]

        return visited
                
    def find_all_clusters(self):
        """
        Find clusters of the percolating species.  
        Results will be saved in self.clusters.

        Using the resulting list of clusters, the  ordering parameter
        P_infinity is calculated and stored in self.p_infinity.

        """
      
        self._clusters     = []
        self._p_infinity   = 0.0
        self._n_cl_max     = 0

        max_size = 0

        done = []
        for i in range(self.num_sites):
            if ((self._decoration[i] == PERCOLATING) 
                and (not i in done)):
                cl = self.get_cluster(i, [])
                self._clusters.append(cl)
                done += cl
                # keep track of the max. cluster size
                n_cl = len(cl)
                max_size = max(max_size, n_cl)

        if (max_size > 0):
            self._p_infinity = float(max_size)/float(self.n_percolating)
            self._n_cl_max   = max_size
            self.calc_susceptibility()

    def find_spanning_cluster(self):
        """
        Count clusters that span the simulation cell.
        (No p.b.c.)
        """
      
        self._spanning_cluster = []

        done = []
        for axis in range(3):
            for i in self._bd_min[axis]:
                if ((self._decoration[i] == PERCOLATING) 
                    and (not i in done)):
                    cl = self.get_cluster(i, [], pbc=False)
                    done += cl
                    for j in self._bd_max[axis]:
                        if j in cl:
                            self._spanning_cluster = cl
                            return 1.0

        return 0.0

            
    #--------------------- derivable quantities -----------------------#

    def calc_susceptibility(self):
        """
        Calculate the percolation susceptibility based on the list
        of clusters found in the system.
        """

        self._susceptibility = 0.0

        for cl in self._clusters:
            n_cl = len(cl)
            if not (n_cl == self._n_cl_max):
                self._susceptibility += float(n_cl*n_cl)

        self._susceptibility /= float(self.p_percolating)

    #------------------------------------------------------------------#
    #                         private methods                          #
    #------------------------------------------------------------------#

    def _build_neighbor_list(self, dr=0.2):
        """
        Determine the list of neighboring sites for 
        each site of the lattice.  Allow the next neighbor
        distance to vary about `dr'.
        """
        
        s = self._structure
        nsites = self.num_sites

        avec = self._structure.lattice.matrix
        coo  = self._structure.frac_coords
        self._nblist = NeighborList(avec, coo)

        """
        Determine nearest neighbor distance for each site.
        Account for numerical errors and relaxations by using 
        a range of dr.
        """
        
        dNN = np.zeros(nsites)
        dNN[:] = s.lattice.a + s.lattice.b + s.lattice.c + 2*dr
        nbs = range(nsites)
        for s1 in range(nsites):
            possible = self._nblist.get_possible_neighbors(s1)
            for s2 in possible:
                (d, jimg) = s[s1].distance_and_image(s[s2])
#                d = s.get_distance(s1,s2)
                if (d <= dNN[s1] + dr):
                    if (d < dNN[s1] - dr):
                        nbs[s1] = []
                    dNN[s1] = min(d, dNN[s1])
                    nbs[s1].append((s2,jimg))
#                    nbs[s1].append(s2)

        self._dNN       = dNN
        self._neighbors = nbs

    def _get_boundaries(self, dr=0.2):
        """
        Determine sites on the boundaries of the lattice cell.
        """

        self._bd_min = range(3)
        self._bd_max = range(3)

        coo = np.array(self._structure.frac_coords)

        # assert that all coordinates are in the [0:1.0[ interval
        assert (np.alltrue(coo >= 0.0) and np.alltrue(coo < 1.0))

        for axis in range(3):
            dr_scal = dr/self._structure.lattice.abc[axis]
            idx = np.argsort(coo[:,axis])
            val_min = coo[idx[0],axis]
            val = val_min
            self._bd_min[axis] = []
            i = 0
            while (val < val_min + dr_scal):
                self._bd_min[axis].append(idx[i])
                i += 1
                val = coo[idx[i],axis]
            val_max = coo[idx[-1],axis]
            val = val_max
            self._bd_max[axis] = []
            i = 1
            while (val > val_max - dr_scal):
                self._bd_max[axis].append(idx[-i])
                i += 1
                val = coo[idx[-i],axis]

#----------------------------------------------------------------------#



class IncompatibleStructureException(Exception):
    """
    Raised, if the input structure is 
    not compatible with the Percolator.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
