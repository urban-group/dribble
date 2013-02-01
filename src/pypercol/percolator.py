from __future__ import print_function

import numpy as np
import sys

from sys         import stderr
from scipy.stats import binom
from pynblist    import NeighborList
from aux         import uprint
from aux         import ProgressBar

EPS   = 100.0*np.finfo(np.float).eps

#----------------------------------------------------------------------#
#                           Percolator class                           #
#----------------------------------------------------------------------#

class Percolator(object):

    def __init__(self, lattice_vectors, frac_coords):
        """
        Arguments:
          lattice_vectors    3x3 matrix with lattice vectors in rows
          frac_coords        Nx3 array; fractional coordinates of the 
                             N lattice sites
        """

        """                    static data

        _avec[i][j]   j-th component of the i-th lattice vector
        _coo[i][j]    j-th component of the coordinates of the i-th 
                      lattice site
        _nsites       total number of lattice sites

        _dNN[i]             nearest neighbor distance from the i-th site
        _neighbors[i][j]    j-th neighbor site of site i
        _T_vectors[i][j]    the translation vector belonging to 
                            _neighbors[i][j]
        """

        self._avec   = np.array(lattice_vectors)
        self._coo    = np.array(frac_coords)
        self._nsites = len(self._coo)

        self.reset()

        self._dNN       = []
        self._neighbors = []
        self._T_vectors = []
        self._build_neighbor_list()

    @classmethod
    def from_structure(cls, structure):
        """
        Create a Percolator instance based on the lattice vectors
        defined in a `structure' object.

        Arguments:
          structure       an instance of pymatgen.core.structure.Structure
          use_decoration  Boolean; use site occupancies of `structure'
          percolating     String; the percolating species
        """
    
        avec   = structure.lattice.matrix
        coo    = structure.frac_coords
        percol = cls(avec, coo)

        return percol

    def reset(self):
        """
        Reset the instance to the state of initialization.
        """

        """              internal dynamic data

        _cluster[i]     ID of cluster that site i belongs to; < 0, if
                        site i is vacant
        _nclusters      total number of clusters with more than 0 sites
                        Note: len(_first) can be larger than this !
        _first[i]       first site (head) of the i-th cluster
        _size[i]        size of cluster i (i.e., number of sites in i)
        _is_spanning[i][j]  True, if cluster i is spanning in direction j
        _next[i]        the next site in the same cluster as site i;
                        < 0, if site i is the final site
        _vec[i][j]      j-th component of the vector that connects
                        site i with the head site of the cluster

        _percolating[i]     i-th percolating site
        _nonpercolating[i]  i-th non-percolating (empty) site

        """

        self._cluster     = np.empty(self._nsites, dtype=int)
        self._cluster[:]  = -1
        self._nclusters   = 0
        self._first       = []
        self._size        = []
        self._is_spanning = []
        self._largest     = -1
        self._next        = np.empty(self._nsites, dtype=int)
        self._next[:]     = -1
        self._vec         = np.zeros(self._coo.shape)

        self._percolating    = []
        self._nonpercolating = range(self._nsites)

    #------------------------------------------------------------------#
    #                            properties                            #
    #------------------------------------------------------------------#

    @property
    def num_clusters(self):
        return self._nclusters

    @property
    def num_percolating(self):
        return len(self._percolating)

    @property
    def num_nonpercolating(self):
        return len(self._nonpercolating)

    @property
    def num_sites(self):
        return self._nsites

    #------------------------------------------------------------------#
    #                          public methods                          #
    #------------------------------------------------------------------#

    def add_percolating_site(self, site=None):
        """
        Change status of SITE to be percolating.
        If SITE is not specified, it will be randomly selected.
        """

        if (self.num_nonpercolating <= 0):
            stderr.write("Warning: all sites are already occupied\n")
            return

        if not site:
            sel = np.random.random_integers(0,len(self._nonpercolating)-1)
            site = self._nonpercolating[sel]
        else:
            sel = self._nonpercolating.index(site)

        del self._nonpercolating[sel]
        self._percolating.append(site)

        # for the moment, add a new cluster
        self._first.append(site)
        self._size.append(1)
        self._is_spanning.append([False, False, False])
        self._cluster[site] = len(self._first) - 1
        self._nclusters += 1
        self._vec[site, :]  = [0.0, 0.0, 0.0]
        if (self._largest < 0):
            self._largest = self._cluster[site]

        # check, if this site
        # - defines a new cluster,
        # - will be added to an existing cluster, or
        # - connects multiple existing clusters.
        for i in xrange(len(self._neighbors[site])):
            nb = self._neighbors[site][i]
            cl = self._cluster[nb]
            if (cl >= 0):
                self._merge_clusters(cl, nb, self._cluster[site], 
                                     site, -self._T_vectors[site][i])
                    
    def calc_p_infinity(self, plist, samples=500, save_discrete=False):
        """
        Calculate a Monte-Carlo estimate for the probability P_inf 
        to find an infinitly extended cluster along with the percolation
        susceptibility Chi.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete.dat)

        Returns:
          tuple (P_inf, Chi), with lists of P_inf and Chi values 
          that correspond to the desired probabilities in `plist'
        """

        uprint(" Calculating P_infty(p) and Chi(p).")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        Pn = np.zeros(self._nsites)
        Xn = np.zeros(self._nsites)
        w  = 1.0/float(samples)
        w2 = w*float(self._nsites)
        for i in xrange(samples):
            pb()
            self.reset()
            for n in xrange(self._nsites):
                self.add_percolating_site()
                Pn[n] += w*(float(self._size[self._largest])/float(self.num_percolating))
                for cl in xrange(len(self._size)):
                    if cl == self._largest:
                        continue
                    Xn[n] += w2*self._size[cl]**2/float(n)

        pb()

        if save_discrete:
            fname = 'discrete.dat'
            uprint("Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in xrange(self._nsites):
                    f.write("{} {} {}\n".format(n+1, Pn[n], Xn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, self._nsites+1, dtype=int)
        Pp = np.empty(len(plist))
        Xp = np.empty(len(plist))
        for i in xrange(len(plist)):
            Pp[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Pn)
            Xp[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Xn)
        
        return (Pp, Xp)

    def calc_p_wrapping(self, plist, samples=500, save_discrete=False):
        """
        Calculate a Monte-Carlo estimate for the probability P_wrap
        to find a wrapping cluster.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete-wrap.dat)

        Returns:
          tuple (P_wrap, P_wrap_c)
          P_wrap         list of values of P_wrap that corespond to `plist'
          P_wrap_c       the cumulative of P_wrap
        """

        uprint(" Calculating P_wrap(p).")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        Pn = np.zeros(self._nsites)
        Pnc = np.zeros(self._nsites)
        w  = 1.0/float(samples)
        for i in xrange(samples):
            pb()
            self.reset()
            for n in xrange(self._nsites):
                self.add_percolating_site()
                spanning = self._is_spanning[self._largest]
                if (np.any(spanning)):
                    Pn[n]   += w
                    Pnc[n:] += w
                    break

        pb()

        if save_discrete:
            fname = 'discrete-wrap.dat'
            uprint("Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in xrange(self._nsites):
                    f.write("{} {}\n".format(n+1, Pn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, self._nsites+1, dtype=int)
        Pp = np.empty(len(plist))
        Ppc = np.empty(len(plist))
        for i in xrange(len(plist)):
            Pp[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Pn)
            Ppc[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Pnc)
        
        return (Pp, Ppc)


    def find_percolation_point(self, samples=500, file_name=None):
        """
        Determine an estimate for the site percolation threshold p_c.
        """

        uprint(" Calculating an estimate for the percolation point p_c.")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        pc_any = 0.0
        pc_two = 0.0
        pc_all = 0.0

        w = 1.0/float(samples)
        for i in xrange(samples):
            pb()
            self.reset()
            done_any = done_two = False
            for n in xrange(self._nsites):
                self.add_percolating_site()
                spanning = self._is_spanning[self._largest]
                if (np.any(spanning)) and not done_any:
                    pc_any += w*float(n)/float(self._nsites)
                    done_any = True
                    if file_name:
                        self.save_cluster(self._largest, 
                             file_name=(file_name+("-1.%05d"%(i,))))
                if (np.sum(np.where(spanning,1,0))>=2) and not done_two:
                    pc_two += w*float(n)/float(self._nsites)
                    done_two = True
                    if file_name:
                        self.save_cluster(self._largest, 
                             file_name=(file_name+("-2.%05d"%(i,))))
                if np.all(spanning):
                    pc_all += w*float(n)/float(self._nsites)
                    if file_name:
                        self.save_cluster(self._largest, 
                             file_name=(file_name+("-3.%05d"%(i,))))
                    break
                if n == self._nsites-1:
                    stderr.write("Error: All sites occupied, but no spanning cluster!?\n")
                    stderr.write("       Have a look at `POSCAR-Error'.\n")
                    self.save_cluster(self._largest, file_name="POSCAR-Error")
                    print(spanning)
                    sys.exit()

        pb()
      
        return (pc_any, pc_two, pc_all)

    def check_if_percolating(self):
        """
        Check, if the largest cluster is percolating.
        """

        cl = self._largest
        cmin = np.zeros(3)
        cmax = np.zeros(3)

        self._is_percolating = [False, False, False]

        i = self._first[cl]
        while self._next[i] >=0:
            i = self._next[i]
            vec = self._vec[i]
            cmin = np.where(cmin <= vec, cmin, vec)
            cmax = np.where(cmax >= vec, cmax, vec)
            self._is_percolating = (cmax - cmin >= 1.0)
            if np.all(self._is_percolating):
                break

    def save_cluster(self, cluster, file_name="CLUSTER"):
        """
        Save a particular cluster to an output file.
        Relies on `pymatgen' for the file I/O.
        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vaspio      import Poscar

        species = ["H" for i in range(self._nsites)]
        i = self._first[cluster]
        species[i] = "C"
        while (self._next[i] >= 0):
            i = self._next[i]
            species[i] = "C"

        species = np.array(species)
        idx = np.argsort(species)

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

    def save_neighbors(self, site, file_name="NEIGHBORS"):
        """
        Save neighbors of site SITE to an output file.
        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vaspio      import Poscar

        species = ["H" for i in range(self._nsites)]
        species[site] = "C"
        for nb in self._neighbors[site]:
            species[nb] = "C"

        species = np.array(species)
        idx = np.argsort(species)

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)


    def get_cluster(self, site, visited=[]):
        """
        Recursively determine all other sites connected to SITE.
        """

        if (self._cluster[site] < 0):
            return visited
        
        visited.append(site)
        for nb in self._neighbors[site]:
            if ((not (nb in visited)) and (self._cluster[nb] >= 0)):
                visited += self.get_cluster(nb, visited)[len(visited):]

        return visited
        

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

    def _merge_clusters(self, cluster1, site1, cluster2, site2, T2):
        """
        Add sites of cluster2 to cluster1.
        """

        # vector from head node of cluster2 to head of cluster1
        v_12 = (self._coo[site2] + T2) - self._coo[site1]
        vec  = self._vec[site1] - v_12 - self._vec[site2]
        
        if (cluster1 == cluster2):
            # if `vec' is different from the stored vector, we have
            # a wrapping cluster
            if abs(vec[0]) > 100*EPS:
                self._is_spanning[cluster1][0] = True
            if abs(vec[1]) > 100*EPS:
                self._is_spanning[cluster1][1] = True
            if abs(vec[2]) > 100*EPS:
                self._is_spanning[cluster1][2] = True
            return

        # add vec to all elements of the second cluster
        # and change their cluster ID
        i = self._first[cluster2]
        self._vec[i, :] += vec
        self._cluster[i] = cluster1
        while (self._next[i] >= 0):
            i = self._next[i]
            self._vec[i,:] += vec
            self._cluster[i] = cluster1

        # insert second cluster right after the head node in 
        # cluster 1
        j = self._first[cluster1]
        self._next[i] = self._next[j]
        self._next[j] = self._first[cluster2]

        # keep track of the cluster sizes and the largest cluster
        self._size[cluster1] += self._size[cluster2]
        if (self._size[cluster1] > self._size[self._largest]):
            self._largest = cluster1

        # keep track of the spanning property
        l1 = self._is_spanning[cluster1]
        l2 = self._is_spanning[cluster2]
        self._is_spanning[cluster1] = [l1[i] or l2[i] for i in range(len(l1))]

        # Only delete the cluster, if it is the last in the list.
        # Otherwise we would have to update the cluster IDs on all sites.
        self._nclusters -= 1
        if (len(self._first) == cluster2+1):
            del self._first[cluster2]
            del self._size[cluster2]
            del self._is_spanning[cluster2]
        else:
            self._first[cluster2] = -1
            self._size[cluster2]  = 0
            self._is_spanning[cluster2] = [False, False, False]

class Percolator2(object):

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

        # the recursive cluster search requires a larger recursion depth
        sys.setrecursionlimit(10000)
            
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

    def random_decoration_n(self, n):
        """
        Randomly decorate lattice with exactly n percolating sites.
        If n greater than the number of sites, all sites will be 
        occupied by the percolating species (boring).
        """
        
        n = min(self.num_sites, n)
        self._decoration[:] = NONPERCOLATING
        for i in range(n):
            self._decoration[i] = PERCOLATING
        self.randomize_decoration()

    def random_decoration_p(self, p):
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
