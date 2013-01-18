"""
Numerical calculaion of the site and bond percolation of an 
arbitrary crystal lattice.
"""

__author__ = "Alexander Urban"
__date__   = "2013-01-15"

import numpy as np
from sys import stderr

PERCOLATING    = 1
NONPERCOLATING = 0

class Percolator:

    def __init__(self, structure, percolating='Li'):
        """
        structure     an instance of pymatgen.core.structure.Structure
        percolating   the percolating atomic species
        """

        self._structure     = structure
        self._neighbors     = []
        self._clusters      = []
        self._p_infinity    = 0.0
        self._n_percolating = 0

        self._build_neighbor_list()

        self._decoration = np.zeros(self.num_sites, dtype=int)
        self.decorate(structure, percolating)

    def __str__(self):
        return str(self._structure.lattice)

    #-------------------------- properties ----------------------------#

    @property
    def clusters(self):
        return self._clusters

    @property
    def decoration(self):
        return self._decoration

    @property
    def num_sites(self):
        return self._structure.num_sites

    @property
    def n_percolating(self):
        return self._n_percolating

    @property
    def p_infinity(self):
        return self._p_infinity

    @property
    def structure(self):
        return self._structure

    #--------------------------- methods ------------------------------#

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

    def get_cluster(self, site, visited=[]):
        """
        Recursively determine all percolating sites connected to SITE.
        """

        if (self._decoration[site] == NONPERCOLATING):
            return visited
        
        visited.append(site)
        for nb in self._neighbors[site]:
            if ((not (nb in visited)) and (self._decoration[nb] == PERCOLATING)):
                visited += self.get_cluster(nb, visited)[len(visited):]

        return visited
                
    def find_all_clusters(self):
        """
        Find clusters of the percolating species.  
        Results will be saved in self.clusters.

        Using the resulting list of clusters, the  ordering parameter
        P_infinity is calculated and stored in self.p_infinity.

        """
      
        self._clusters = []

        max_size = 0

        done = []
        for i in range(self.num_sites):
            if ((self._decoration[i] == PERCOLATING) 
                and (not i in done)):
                cl = self.get_cluster(i, [])
                max_size = max(max_size, len(cl))
                self._clusters.append(cl)
                done += cl

        if (max_size > 0):
            self._p_infinity = float(max_size)/float(self.n_percolating)
        else:
            self._p_infinity = 0.0
      
    def get_neighbors(self, site_index=0):
        return self._neighbors[site_index]

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

    #------------------------- private methods ------------------------#

    def _build_neighbor_list(self):
        """
        Determine the list of neighboring sites for 
        each site of the lattice.
        """
        
        s = self._structure
        nsites = self.num_sites

        """
        Determine nearest neighbor distance for each site.
        Account for numerical errors and relaxations by using a range 
        of dr = 0.2 Angstroms.
        """
        dr  = 0.2
        dNN = np.zeros(nsites)
        dNN[:] = np.max((s.lattice.a, s.lattice.b, s.lattice.c)) + 2*dr
        nbs = range(nsites)
        for s1 in range(nsites):
            for s2 in range(nsites):
                if (s1 == s2):
                    continue
                d = s.get_distance(s1,s2)
                if (d <= dNN[s1] + dr):
                    if (d < dNN[s1] - dr):
                        nbs[s1] = []
                    dNN[s1] = min(dNN[s1],d)
                    nbs[s1].append(s2)

        self._neighbors = nbs




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
