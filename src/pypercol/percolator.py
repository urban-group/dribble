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

        _special      True, if special percolation rules have been defined

        _dNN[i]             nearest neighbor distance from the i-th site
        _neighbors[i][j]    j-th neighbor site of site i
        _bonds[i][j]        True, if there is a bond between the j-th site 
                            and its j-th neighbor
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
        self._nsites = len(self._coo)

        # self.reset()

        self._special   = False

        self._dNN       = []
        self._neighbors = []
        self._bonds     = []
        self._T_vectors = []
        self._build_neighbor_list()

        # max. number of bonds is half the number of nearest neighbors
        self._nbonds_tot = 0
        for i in xrange(self._nsites):
            nbs = len(set(self._neighbors[i]))
            self._nbonds_tot += nbs
            self._bonds.append(np.array(nbs*[False]))
        self._nbonds_tot /= 2

        self.reset()

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

    def reset(self):
        """
        Reset the instance to the state of initialization.
        """

        """              internal dynamic data

        _cluster[i]     ID of cluster that site i belongs to; < 0, if
                        site i is vacant
        _nclusters      total number of clusters with more than 0 sites
                        Note: len(_first) can be larger than this !
        _nbonds         the number of bonds found so far
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
        self._nbonds      = 0
        self._first       = []
        self._size        = []
        self._is_spanning = []
        self._largest     = -1
        self._next        = np.empty(self._nsites, dtype=int)
        self._next[:]     = -1
        self._vec         = np.zeros(self._coo.shape)

        self._percolating    = []
        self._nonpercolating = range(self._nsites)

        for i in range(self._nsites):
            self._bonds[i][:] = False

    #------------------------------------------------------------------#
    #                            properties                            #
    #------------------------------------------------------------------#

    @property
    def largest_cluster(self):
        return self._largest

    @property
    def neighbors(self):
        return self._neighbors

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

    def get_cluster_of_site(self, site, visited=[]):
        """
        Recursively determine all other sites connected to SITE.
        """

        if (self._cluster[site] < 0):
            return visited
        
        visited.append(site)
        for nb in self._neighbors[site]:

            if ((not (nb in visited)) 
                and (self._cluster[nb] >= 0) 
                and self._check_special(site,nb)):

                visited += self.get_cluster_of_site(nb, visited)[len(visited):]

        return visited

    def get_common_neighbors(self, site1, site2):
        """
        Returns a list of common neighbor sites of SITE1 and SITE2.
        """

        return list(set(self._neighbors[site1]) 
                    & set(self._neighbors[site2]))

    def set_special_percolation_rule(self, num_common=0):
        """
        Define special criteria for a bond between two occupied sites 
        to be percolating.

        Arguments:
          num_common   number of common neighbors of two sites i and j
                       that have to be occupied to form a percolating
                       bond between i and j.
        """

        def new_special(site1, site2):
            """
            Check, if the special percolation rule is fulfilled
            between sites SITE1 and SITE2.
            
            This instance does indeed define a special rule.
            """

            percolating = False

            occupied = 0
            common_neighbors = self.get_common_neighbors(site1, site2)

            for nb in common_neighbors:
                if self._cluster[nb] >= 0:
                    occupied += 1
                if occupied >= num_common:
                    percolating = True
                    break

            return percolating

        self._special       = True
        self._check_special = new_special


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

                # update also next nearest neighbors
                # (only in case of special percolation rules)
                if not self._special:
                    continue

                # loop over the neighbors of the neighbor
                for j in xrange(len(self._neighbors[nb])):
                    nb2 = self._neighbors[nb][j]
                    cl2 = self._cluster[nb2]
                    if (cl2 >= 0):
                        self._merge_clusters(cl2, nb2, self._cluster[nb], 
                                             nb, -self._T_vectors[nb][j])

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
        w2 = w*self._nsites
        for i in xrange(samples):
            pb()
            self.reset()
            for n in xrange(self._nsites):
                self.add_percolating_site()
                spanning = self._is_spanning[self._largest]
                if (np.any(spanning)):
                    Pnc[n:] += w
                    Pn[n]   += w2
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

    def percolating_bonds(self, plist, samples=500, save_discrete=False):
        """
        Estimate number of percolating bonds in dependence of site
        concentration.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete-wrap.dat)

        Returns:
          list of fractions of all bonds for certain concentration
        """

        uprint(" Calculating fraction F_bonds(p) of percolating bonds.")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        Pn = np.zeros(self._nsites)
        w  = 1.0/float(samples)/float(self._nbonds_tot)
        for i in xrange(samples):
            pb()
            self.reset()
            for n in xrange(self._nsites):
                self.add_percolating_site()
                Pn[n]   += w*float(self._nbonds)
        pb()

        if save_discrete:
            fname = 'discrete-bonds.dat'
            uprint("Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in xrange(self._nsites):
                    f.write("{} {}\n".format(n+1, Pn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, self._nsites+1, dtype=int)
        Pp = np.empty(len(plist))
        for i in xrange(len(plist)):
            Pp[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Pn)
        
        return Pp
        
    def find_percolation_point(self, samples=500, file_name=None):
        """
        Determine an estimate for the site percolation threshold p_c.
        """

        uprint(" Calculating an estimate for the percolation point p_c.")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        pc_site_any = 0.0
        pc_site_two = 0.0
        pc_site_all = 0.0

        pc_bond_any = 0.0
        pc_bond_two = 0.0
        pc_bond_all = 0.0

        w1 = 1.0/float(samples)/float(self._nsites)
        w2 = 1.0/float(samples)/float(self._nbonds_tot)
        for i in xrange(samples):
            pb()
            self.reset()
            done_any = done_two = False
            for n in xrange(self._nsites):
                self.add_percolating_site()
                spanning = self._is_spanning[self._largest]
                if (np.any(spanning)) and not done_any:
                    pc_site_any += w1*float(n)
                    pc_bond_any += w2*float(self._nbonds)
                    done_any = True
                    if file_name:
                        self.save_cluster(self._largest, 
                             file_name=(file_name+("-1.%05d"%(i,))))
                if (np.sum(np.where(spanning,1,0))>=2) and not done_two:
                    pc_site_two += w1*float(n)
                    pc_bond_two += w2*float(self._nbonds)
                    done_two = True
                    if file_name:
                        self.save_cluster(self._largest, 
                             file_name=(file_name+("-2.%05d"%(i,))))
                if np.all(spanning):
                    pc_site_all += w1*float(n)
                    pc_bond_all += w2*float(self._nbonds)
                    if file_name:
                        self.save_cluster(self._largest, 
                             file_name=(file_name+("-3.%05d"%(i,))))
                    break
                if n == self._nsites-1:
                    stderr.write(
                        "Error: All sites occupied, but no spanning cluster!?"
                        + "\n       "
                        + "Maybe you defined a percolation rule that never percolates."
                        + "\n       Have a look at `POSCAR-Error'.\n")
                    self.save_cluster(self._largest, file_name="POSCAR-Error")
                    print(spanning)
                    sys.exit()

        pb()
      
        return (pc_site_any, pc_site_two, pc_site_all,
                pc_bond_any, pc_bond_two, pc_bond_all)


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

        if not self._check_special(site1, site2):
            return
        
        # remember bonds
        nb1 = self._neighbors[site1].index(site2)
        if not self._bonds[site1][nb1]:
            nb2 = self._neighbors[site2].index(site1)
            self._bonds[site1][nb1] = True
            self._bonds[site2][nb2] = True
            self._nbonds += 1

        # vector from head node of cluster2 to head of cluster1
        v_12 = (self._coo[site2] + T2) - self._coo[site1]
        vec  = self._vec[site1] - v_12 - self._vec[site2]
        
        if (cluster1 == cluster2):
            # if `vec' is different from the stored vector, we have
            # a wrapping cluster, i.e. we found the periodic image of 
            # a site that is already part of the cluster
            if abs(vec[0]) > 0.5:
                self._is_spanning[cluster1][0] = True
            if abs(vec[1]) > 0.5:
                self._is_spanning[cluster1][1] = True
            if abs(vec[2]) > 0.5:
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

    def _check_special(self, site1, site2):
        """
        Check, if the special percolation rule is fulfilled
        between sites SITE1 and SITE2.

        However, this instance does not define any special rule.
        """

        # This method may be replaced at run-time by calling
        # set_special_percolation_rule().

        return True

#----------------------------------------------------------------------#
#                              exceptions                              #
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
