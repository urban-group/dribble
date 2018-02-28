""" Copyright (c) 2013-2017 Alexander Urban

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function, division, unicode_literals

import numpy as np
import sys

from sys import stderr
from scipy.stats import binom

from .misc import uprint
from .misc import ProgressBar
from .lattice import Lattice
from .sublattice import Bond

__autor__ = "Alexander Urban"
__date__ = "2013-02-15"

EPS = 100.0*np.finfo(np.float).eps


class Percolator(object):
    """
    The Percolator class implements a fast MC algorithm for percolation
    analysis on regular lattices [1].  Several different methods allow
    the computation of various quantities related to percolation, such
    as the percolation susceptibility, the ratio of the largest cluster
    of sites to all occupied sites, the probability for wrapping
    (periodic) clusters vs. the site concentration, and the percolation
    threshold.

    See the doc string of the individual methods for details:

       calc_p_infinity
       calc_p_wrapping
       percolating_bonds
       inaccessible_sites
       percolation_point

    Apart from regular percolation analyses, special percolation rules
    can be specified using `set_special_percolation_rule()'.  This
    allows to modify the criteria of percolating bonds.

    [1] Newman and Ziff, Phys. Rev. Lett. 85, 4104-4107 (2000).

    """

    def __init__(self, lattice, percolating_species=[],
                 static_species=[], initial_concentrations=None,
                 site_rules=None, bond_rules=None):
        """
        Arguments:
          lattice              an instance of the Lattice class
          percolating_species  list of species associated with occupied sites
          static_species       list of species associated with static sites
          initial_concentrations  dictionary with species concentrations
                               for each sublattice

        """

        """                    static data

        _bonds[i][j]        True, if there is a bond between the j-th site
                            and its j-th neighbor
        _nsurface           number of sites at cell boundary
                            (computed, if needed)
        _T_vectors[i][j]    the translation vector belonging to
                            _neighbors[i][j]
        _nbonds_tot         maximum number of possible bonds between the sites
        cluster[i]          ID of cluster that site i belongs to; < 0, if
                            site i is vacant
        """

        self.lattice = lattice
        self._percolating_species = percolating_species
        self._static_species = static_species
        self._initial_concentrations = initial_concentrations
        self.site_rules = {} if site_rules is None else site_rules
        self.bond_rules = {} if bond_rules is None else bond_rules

        # references for convenient access (no copies)
        self._avec = self.lattice._avec
        self._coo = self.lattice._coo
        self._nsites = self.lattice._nsites
        self.neighbors = self.lattice._nn
        self._T_vectors = self.lattice._T_nn
        self._nsurface = self.lattice._nsurface
        self.cluster = self.lattice._occup
        self._occupied = self.lattice._occupied
        self._vacant = self.lattice._vacant

        self._bonds = []
        # max. number of bonds is half the number of nearest neighbors
        self._nbonds_tot = 0
        for i in range(self._nsites):
            nbs = len(self.neighbors[i])
            if nbs > len(set(self.neighbors[i])):
                raise ValueError("Cell size too small. Try supercell.")
            self._nbonds_tot += nbs
            self._bonds.append(np.array(nbs*[False]))
        self._nbonds_tot /= 2

        initially_occupied_sites = lattice._occupied[:]
        self.reset(occupied=initially_occupied_sites)

        sys.setrecursionlimit(50000)

        self.progress_bar_char = u"\u25ae"

    @classmethod
    def from_structure(cls, structure, **kwargs):
        """
        Create a Percolator instance based on the lattice vectors
        defined in a `structure' object.

        Arguments:
          structure       an instance of pymatgen.core.structure.Structure
          all keyword arguments are passed to the Lattice class constructor
        """

        avec = structure.lattice.matrix
        coo = structure.frac_coords
        lattice = Lattice(avec, coo, **kwargs)
        percol = cls(lattice)

        return percol

    @classmethod
    def from_coordinates(cls, lattice_vectors, frac_coords, **kwargs):
        """
        Create a Percolator instance based on just a set of lattice vectors
        and fractional coordinates.

        Arguments:
          lattice_vectors (array)   Lattice vectors in rows
          frac_coords (array)       Fractional coordinates
          all keyword arguments of the main constructor
        """

        lattice = Lattice(lattice_vectors, frac_coords)
        percol = cls(lattice, **kwargs)

        return percol

    @classmethod
    def from_input_object(cls, inp, lattice, verbose=False, **kwargs):
        """
        Create a Percolator instance based on an Input Parameters object.

        Arguments:
          inp (percol.input.Input): Input Data object
          lattice (percol.lattice.Lattice): Periodic lattice

        """
        site_rules = {sl: inp.sublattices[sl].site_rules
                      for sl in inp.sublattices}
        bond_rules = {bond: inp.bonds[bond].bond_rules
                      for bond in inp.bonds}
        percolator = cls(lattice,
                         percolating_species=inp.percolating_species,
                         static_species=inp.static_species,
                         initial_concentrations=inp.initial_occupancy,
                         site_rules=site_rules,
                         bond_rules=bond_rules)
        return percolator

    def reset(self, occupied=[]):
        """
        Reset the instance to the state of initialization.
        Static occupied sites are ensured to be always occupied.

        """

        """              internal dynamic data

        cluster[i]      ID of cluster that site i belongs to; < 0, if
                        site i is vacant
        _nclusters      total number of clusters with more than 0 sites
                        Note: len(_first) can be larger than this !
        _nbonds         the number of bonds found so far
        _npercolating   number of sites that are members of percolating
                        clusters
        _nclus_percol   number of percolating clusters
        _npaths         total number of percolation paths
        _first[i]       first site (head) of the i-th cluster
        _size[i]        size of cluster i (i.e., number of sites in i)
        _is_wrapping[i][j]  number of times that cluster i is wrapping in
                        direction j
        _next[i]        the next site in the same cluster as site i;
                        < 0, if site i is the final site
        _vec[i][j]      j-th component of the vector that connects
                        site i with the head site of the cluster

        _occupied[i]    i-th occupied site
        _vacant[i]      i-th vacant site

        """

        self._nclusters = 0
        self._nbonds = 0
        self._npercolating = 0
        self._nclus_percol = 0
        self._npaths = 0
        self._first = []
        self._size = []
        self._is_wrapping = []
        self._next = np.empty(self._nsites, dtype=int)
        self._next[:] = -1
        self._vec = np.zeros(self._coo.shape)

        self.largest_cluster = -1

        # initial random species decoration, if specified
        species = [None for i in occupied]
        if len(occupied) == 0 and self._initial_concentrations is not None:
            self.lattice.random_species_decoration(
                self._initial_concentrations,
                occupying_species=self._percolating_species,
                static=self._static_species)
            occupied = self.lattice._occupied[:]
            species = [self.lattice.species[i] for i in occupied]

        # completely reset lattice
        self.cluster[:] = -1
        self._occupied = []
        self._vacant = list(range(self._nsites))

        # Composition of all clusters and total composition of percolating
        # clusters
        self._cluster_comp = {}
        self._percol_comp = {}

        for i in range(self._nsites):
            self._bonds[i][:] = False

        # repopulate initial occupations, if any:
        for i, s in enumerate(occupied):
            self.add_percolating_site(site=s, species=species[i])

    def __str__(self):
        ostr = "\n An instance of the Percolator class\n\n"
        ostr += " Structure info:\n"
        ostr += str(self.lattice)
        return ostr

    def __repr__(self):
        return self.__str__()

    @property
    def num_clusters(self):
        return self._nclusters

    @property
    def num_percolating(self):
        return len(self._percolating)

    @property
    def num_sites(self):
        return self.lattice.num_sites

    @property
    def num_sites_not_static(self):
        return self.lattice.num_sites_not_static

    @property
    def _not_static(self):
        return self.lattice.not_static

    @property
    def num_vacant(self):
        return self.lattice.num_vacant

    @property
    def num_occupied(self):
        return self.lattice.num_occupied

    @property
    def num_vacant_not_static(self):
        return self.lattice.num_vacant_not_static

    @property
    def num_occupied_not_static(self):
        return self.lattice.num_occupied_not_static

    @property
    def _vacant_not_static(self):
        return self.lattice.vacant_not_static

    @property
    def _occupied_not_static(self):
        return self.lattice.occupied_not_static

    @property
    def _static_occupied(self):
        return self.lattice.static_occupied

    @property
    def _static_vacant(self):
        return self.lattice.static_vacant

    def get_cluster_of_site(self, site, vec=[0, 0, 0], visited=[]):
        """
        Recursively determine all other sites connected to SITE.
        """

        if (self.cluster[site] < 0):
            # vacant site (why are we here?)
            return visited

        nspanning = np.array([0, 0, 0], dtype=int)
        newsites = [site]

        for i in range(len(self.neighbors[site])):
            nb = self.neighbors[site][i]
            # neighboring site occupied and bound?
            if ((self.cluster[nb] >= 0) and self.check_bond(site, nb)):
                # vector to origin
                T = self._T_vectors[site][i]
                v_12 = (self._coo[nb] + T) - self._coo[site]
                vec_nb = vec + v_12
                if not (nb in visited+newsites):
                    self._vec[nb] = vec_nb
                    (subcluster, subspan
                     ) = self.get_cluster_of_site(nb, vec_nb, visited+newsites)
                    newsites += subcluster
                    nspanning += subspan
                else:
                    # nb has already been visited,
                    # was it a different image?
                    diff = np.abs(self._vec[nb] - vec_nb)
                    nspanning += np.where(diff > 0.5, 1, 0)

        return (newsites, nspanning)

    def check_spanning(self, verbose=False, save_clusters=False):
        """
        Check, how many (if any) spanning clusters are present.

        Arguments:
          verbose (bool): if True, print information about clusters to
            standard out
          save_clusters (bool): if True, save a structure file named
            'cluster_{}.vasp' for each detected cluster with more than
            1 site

        """

        visited = []
        nspanning = 0
        avg_size = 0
        num_clusters = 0

        for i in range(self._nsites):
            if (self.cluster[i] > 0) and not (i in visited):
                (cluster, spanning) = self.get_cluster_of_site(i)
                visited += cluster
                if np.sum(spanning) > 0:
                    nspanning += len(cluster)
                num_clusters += 1
                avg_size += len(cluster)
                if verbose:
                    if np.sum(spanning) > 0:
                        uprint("   Spanning cluster with " +
                               "{} sites, ".format(len(cluster)) +
                               "paths in (x,y,z): " +
                               "{} {} {}".format(*spanning))
                    else:
                        uprint("   Isolated cluster with " +
                               "{} sites, ".format(len(cluster)))
                if save_clusters and len(cluster) > 1:
                    self.save_cluster(cluster,
                                      file_name="cluster_{}.vasp".format(i),
                                      sort_species=False)

        if num_clusters > 0:
            avg_size /= num_clusters
        if verbose:
            uprint("")
            uprint(" Average cluster size: " + "{}".format(avg_size))
            uprint("")

        return nspanning

    def get_common_neighbors(self, site1, site2):
        """
        Returns a list of common neighbor sites of SITE1 and SITE2.
        """

        return [nb for nb in self.neighbors[site1]
                if nb in self.neighbors[site2]]

    def set_special_percolation_rule(self, num_common=0, same=None,
                                     require_NN=False, inverse=False,
                                     dr=0.1):
        """
        Define special criteria for a bond between two occupied sites
        to be percolating.

        Arguments:
          num_common   number of common neighbors of two sites i and j
                       that have to be occupied to form a percolating
                       bond between i and j.
          same         0, 1, or 2 --> require same x, y, or z coordinate
                       for the bonding sites; e.g., same=2 will require
                       both sites to be in the same z-layer
          require_NN   if True, the 'num_common' criterion additionally
                       requires the common nearest neighbors themselves
                       to be nearest neighbors (useful for diffusion on
                       an fcc lattice)
          inverse      invert the meaning of num_common so that the
                       common neighbor sites are vacant instead of occupied
          dr           required precision for two coordinates to be the 'same'
        """

        if (require_NN and (num_common < 2)):
            stderr.write("Error: the percolation rule does not make sense!\n")
            stderr.write("       {} common neighbors can not be required\n")
            stderr.write("       to be nearest neighbors themselves!\n")
            sys.exit()

        def new_special(site1, site2):
            """
            Check, if the special percolation rule is fulfilled
            between sites SITE1 and SITE2.

            This instance defines a special rule.

            """

            common_rule = True
            same_rule = True
            NN_rule = True

            if same and (0 <= same <= 2):
                if (abs(self._coo[site1][same]-self._coo[site2][same]) <= dr):
                    same_rule = True
                else:
                    same_rule = False

            if (num_common > 0) and same_rule:
                common_rule = False
                common_nb = self.get_common_neighbors(site1, site2)
                occupied = []
                noccup = 0
                for nb in common_nb:
                    if inverse:
                        if (self.cluster[nb] < 0):
                            occupied.append(nb)
                            noccup += 1
                    else:
                        if (self.cluster[nb] >= 0):
                            occupied.append(nb)
                            noccup += 1
                    if (noccup >= num_common):
                        common_rule = True

            if (require_NN and common_rule and same_rule):
                NN_rule = False
                for i in range(noccup):
                    site1 = occupied[i]
                    nb1 = self.neighbors[site1]
                    num_NN = 1
                    for j in range(i+1, noccup):
                        site2 = occupied[j]
                        if site2 in nb1:
                            num_NN += 1
                            if (num_NN >= num_common):
                                NN_rule = True
                                break

            percolating = (common_rule and same_rule and NN_rule)
            return percolating

        self.check_bond = new_special

    def set_complex_percolation_rule(self, site_rules={}, bond_rules={},
                                     verbose=False):
        """
        Define complex criteria based on site and bond stability criteria.
        The format for these specifications is defined elsewhere.

        Arguments:
          site_rules   dictionary with site stability criteria
          bond_rules   dictionary with bond stability criteria
          verbose      if True, print information to stdout

        """

        def _check_species(sites, rules):
            """
            Check if required min/max species counts are satisfied.
            """
            satisfied = True
            species = [self.lattice.species[s] for s in sites]
            for rule in rules:
                min_required = rule["min"] if "min" in rule else 0
                max_allowed = rule["max"] if "max" in rule else np.inf
                num_sites = len([s for s in species if s in rule["species"]])
                if (num_sites < min_required) or (num_sites > max_allowed):
                    satisfied = False
                    break
            return satisfied

        def _check_stable(site):
            """
            Check whether a site obeys the stability criterion.
            """
            stable = True
            sublattice = self.lattice.site_labels[site]
            for env in site_rules[sublattice]["stable_neighbor_shells"]:
                # loop over neighbor shells in site environment
                for i, nbshell in enumerate(env):
                    nb_list = self.lattice._nbshells[site][i]
                    # loop over sublattices in current neighbor shell
                    for sl in nbshell:
                        # determine neighbors on select sublattice
                        neighbors_on_sl = []
                        for s in nb_list:
                            if self.lattice.site_labels[s] == sl:
                                neighbors_on_sl.append(s)
                        stable &= _check_species(neighbors_on_sl, nbshell[sl])
            return stable

        def new_special(site1, site2):
            """
            Check, d between sites SITE1 and
            SITE2.

            This instance defines a complex percolation rule.

            """

            percolating = True

            # sublattices
            sl1 = self.lattice.site_labels[site1]
            sl2 = self.lattice.site_labels[site2]
            bond = Bond(sl1, sl2)
            if bond not in bond_rules:
                percolating = False
                return

            percolating &= _check_stable(site1)
            percolating &= _check_stable(site2)

            return percolating

        self._complex = True
        self.check_bond = new_special

    def add_percolating_site(self, site=None, species=None):
        """
        Change status of SITE to be percolating.

        Args:
          site    optional site index; if not specified, the site will be
                  randomly selected
          species optional species that will be assigned with the newly
                  percolating site

        Note: The present implementation only considers bonds between
              nearest-neighbor sites!

        """

        if (self.num_vacant_not_static <= 0):
            stderr.write("Warning: all sites are already occupied\n")
            return

        if site is None:
            sel = np.random.random_integers(0, self.num_vacant_not_static-1)
            site = self._vacant_not_static[sel]
        elif (site in self._vacant) and (site not in self._static_vacant):
            sel = self._vacant.index(site)
        else:
            stderr.write("Warning: Attempt to occupy invalid site.\n")
            if site in self._occupied:
                stderr.write(
                    "         Site {} is already occupied.\n".format(site))
            if site in self._static_vacant:
                stderr.write(
                    "         Site {} is a static vacancy.\n".format(site))
            return

        del self._vacant[sel]
        self._occupied.append(site)
        self.lattice.species[site] = species

        # for the moment, add a new cluster
        self._first.append(site)
        self._size.append(1)
        self._is_wrapping.append(np.array([0, 0, 0]))
        cl = len(self._first) - 1
        self.cluster[site] = cl
        self._cluster_comp[cl] = {species: 1}
        self._nclusters += 1
        self._vec[site, :] = [0.0, 0.0, 0.0]
        if (self.largest_cluster < 0):
            self.largest_cluster = self.cluster[site]

        # check, if this site
        # - defines a new cluster,
        # - will be added to an existing cluster, or
        # - connects multiple existing clusters.
        for i in range(len(self.neighbors[site])):
            nb = self.neighbors[site][i]
            cl = self.cluster[nb]
            if (cl >= 0):
                # only consider bonds between nearest neighbors
                if nb in self.lattice._nbshells[site][0]:
                    self._merge_clusters(cl, nb, self.cluster[site],
                                         site, -self._T_vectors[site][i])

                # update also next nearest neighbors
                # loop over the neighbors of the neighbor
                for j in range(len(self.neighbors[nb])):
                    nb2 = self.neighbors[nb][j]
                    cl2 = self.cluster[nb2]
                    # also here: consider only neares-neighbor bonds
                    if (cl2 >= 0) and (nb2 in self.lattice._nbshells[nb][0]):
                        self._merge_clusters(cl2, nb2, self.cluster[nb],
                                             nb, -self._T_vectors[nb][j])

    def calc_p_infinity(self, plist, sequence, samples=500,
                        save_discrete=False):
        """
        Calculate a Monte-Carlo estimate for the probability P_inf
        to find an infinitly extended cluster along with the percolation
        susceptibility Chi.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          sequence       list with sequence of species to be flipped; each
                         pair of species has to be one non-percolating and
                         one percolating species
                         Example:
                         sequence = [["M1", "Li"], ["M2", "Li"]]
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete.dat)

        Returns:
          tuple (P_inf, Chi), with lists of P_inf and Chi values
          that correspond to the desired probabilities in `plist'
        """

        self.reset()
        uprint(" Calculating P_infty(p) and Chi(p).")
        uprint(" Initial composition: ", end="")
        comp = self.lattice.composition
        for s in comp:
            uprint("{} {:.2f} ".format(s, comp[s]), end="")
        uprint("\n Averaging over {} samples:\n".format(samples))
        pb = ProgressBar(samples)

        # Cluster sizes should be based on only the currently active
        # species, but the way it is presently implemented all percolating
        # sites contribute to the cluster size.
        uprint(" Warning: this method does not consider the correct")
        uprint("          cluster sizes if multiple percolating")
        uprint("          species exist.")

        num_active_sites = 0
        for initial, final in sequence:
            num_active_sites += len(self.lattice.sites_of_species(initial))

        Pn = [0 for i in range(num_active_sites)]
        Xn = [0 for i in range(num_active_sites)]
        w = 1.0/float(samples)
        w2 = w*float(num_active_sites)
        for i in range(samples):
            pb()
            self.reset()
            flip_list = []
            for initial, final in sequence:
                sites = self.lattice.sites_of_species(initial)
                np.random.shuffle(sites)
                flip_list += [(s, final) for s in sites]
            if len(flip_list) > num_active_sites:
                for n in range(len(flip_list)-num_active_sites):
                    Pn.append(0)
                    Xn.append(0)
                num_active_sites = len(flip_list)
            for n, (site, species) in enumerate(flip_list):
                self.add_percolating_site(site=site, species=species)
                Pn[n] += w*(float(self._size[self.largest_cluster])/float(n+1))
                for cl in range(len(self._size)):
                    if cl == self.largest_cluster:
                        continue
                    Xn[n] += w2*self._size[cl]**2/float(n+1)
        Pn = np.array(Pn)
        Xn = np.array(Xn)

        pb()

        if save_discrete:
            fname = 'discrete.dat'
            uprint(" Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in range(num_active_sites):
                    f.write("{} {} {}\n".format(n+1, Pn[n], Xn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, num_active_sites+1, dtype=int)
        Pp = np.empty(len(plist))
        Xp = np.empty(len(plist))
        for i in range(len(plist)):
            Pp[i] = np.sum(binom.pmf(nlist, num_active_sites, plist[i])*Pn)
            Xp[i] = np.sum(binom.pmf(nlist, num_active_sites, plist[i])*Xn)

        return (Pp, Xp)

    def calc_p_wrapping(self, plist, sequence, samples=500,
                        save_discrete=False):
        """
        Calculate a Monte-Carlo estimate for the probability P_wrap
        to find a wrapping cluster.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          sequence       list with sequence of species to be flipped; each
                         pair of species has to be one non-percolating and
                         one percolating species
                         Example:
                         sequence = [["M1", "Li"], ["M2", "Li"]]
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete-wrap.dat)

        Returns:
          tuple (P_wrap, P_wrap_c)
          P_wrap         list of values of P_wrap that corespond to `plist'
          P_wrap_c       the cumulative of P_wrap

        """

        self.reset()
        uprint(" Calculating P_wrap(p).")
        uprint(" Averaging over {} samples:\n".format(samples))
        pb = ProgressBar(samples)

        num_active_sites = 0
        for initial, final in sequence:
            num_active_sites += len(self.lattice.sites_of_species(initial))

        Pn = [0 for i in range(num_active_sites)]
        Pnc = [0 for i in range(num_active_sites)]
        w = 1.0/float(samples)
        w2 = w*(num_active_sites)
        for i in range(samples):
            pb()
            self.reset()
            flip_list = []
            for initial, final in sequence:
                sites = self.lattice.sites_of_species(initial)
                np.random.shuffle(sites)
                flip_list += [(s, final) for s in sites]
            if len(flip_list) > num_active_sites:
                for n in range(len(flip_list)-num_active_sites):
                    Pn.append(0)
                    Pnc.append(0)
                num_active_sites = len(flip_list)
            for n, (site, species) in enumerate(flip_list):
                self.add_percolating_site(site=site, species=species)
                wrapping = np.sum(self._is_wrapping[self.largest_cluster])
                if (wrapping > 0):
                    Pnc[n:] += w
                    Pn[n] += w2
                    break
        Pn = np.array(Pn)
        Pnc = np.array(Pnc)

        pb()

        if save_discrete:
            fname = 'discrete-wrap.dat'
            uprint("Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in range(num_active_sites):
                    f.write("{} {}\n".format(n+1, Pn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, num_active_sites+1, dtype=int)
        Pp = np.empty(len(plist))
        Ppc = np.empty(len(plist))
        for i in range(len(plist)):
            Pp[i] = np.sum(binom.pmf(
                nlist, num_active_sites, plist[i])*Pn)
            Ppc[i] = np.sum(binom.pmf(
                nlist, num_active_sites, plist[i])*Pnc)

        return (Pp, Ppc)

    def inaccessible_sites(self, plist, sequence, species, samples=500,
                           save_discrete=False):
        """
        Estimate the number of inaccessible sites, i.e. sites that are
        not part of a percolating cluster, for a given range of
        concentrations.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          sequence       list with sequence of species to be flipped; each
                         pair of species has to be one non-percolating and
                         one percolating species
                         Example:
                         sequence = [["M1", "Li"], ["M2", "Li"]]
          species        reference species whose sites shall be considered;
                         must be a percolating species; usually one of the
                         'final' species from the slipping sequence
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete-wrap.dat)

        Returns:
          list of values corresponding to probabilities in `plist'
        """

        uprint(" Calculating fraction of inaccessible "
               "{} sites.".format(species))
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        num_active_sites = 0
        for initial, final in sequence:
            num_active_sites += len(self.lattice.sites_of_species(initial))

        Pn = [0 for i in range(num_active_sites)]
        Qn = [0 for i in range(num_active_sites)]
        w = 1.0/float(samples)
        for i in range(samples):
            pb()
            self.reset()
            flip_list = []
            for initial, final in sequence:
                sites = self.lattice.sites_of_species(initial)
                np.random.shuffle(sites)
                flip_list += [(s, final) for s in sites]
            if len(flip_list) > num_active_sites:
                for n in range(len(flip_list)-num_active_sites):
                    Pn.append(0)
                    Qn.append(0)
                num_active_sites = len(flip_list)
            for n, (site, species) in enumerate(flip_list):
                self.add_percolating_site(site=site, species=species)
                N_ref = len(self.lattice.sites_of_species(species))
                try:
                    N_ref_percol = self._percol_comp[species]
                except KeyError:
                    N_ref_percol = 0
                Pn[n] += w*float(N_ref - N_ref_percol)/float(N_ref)
                # Pn[n] += w*float(n+1-self._npercolating)/float(n+1)
                Qn[n] += w*float(self._nclus_percol)/float(self._nclusters)
        Pn = np.array(Pn)
        Qn = np.array(Qn)

        pb()

        if save_discrete:
            fname = 'discrete-inaccessible.dat'
            uprint(" Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in range(num_active_sites):
                    f.write("{} {} {}\n".format(n+1, Pn[n], Qn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, num_active_sites+1, dtype=int)
        Pp = np.empty(len(plist))
        Qp = np.empty(len(plist))
        for i in range(len(plist)):
            Pp[i] = np.sum(binom.pmf(nlist, num_active_sites, plist[i])*Pn)
            Qp[i] = np.sum(binom.pmf(nlist, num_active_sites, plist[i])*Qn)

        return (Pp, Qp)

    def percolation_point(self, sequence, samples=500, file_name=None):
        """
        Determine an estimate for the site percolation threshold p_c.

        Args:
          sequence   list with sequence of species to be flipped; each
                     pair of species has to be one non-percolating and
                     one percolating species
                     Example:
                     sequence = [["M1", "Li"], ["M2", "Li"]]
          samples    number of runs to average over
          file_name  file name to store raw data

        """

        self.reset()
        uprint(" Calculating an estimate for the percolation point p_c.")
        uprint(" Initial composition: ", end="")
        comp = self.lattice.composition
        for s in comp:
            uprint("{} {:.2f} ".format(s, comp[s]), end="")
        uprint("\n Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples, char=self.progress_bar_char)

        pc_site_any = 0.0
        pc_site_two = 0.0
        pc_site_all = 0.0

        pc_bond_any = 0.0
        pc_bond_two = 0.0
        pc_bond_all = 0.0

        percolating_composition = {s: 0.0 for s in comp}
        w = 1.0/float(samples)
        w2 = w/float(self._nbonds_tot)
        for i in range(samples):
            pb()
            self.reset()
            flip_list = []
            for initial, final in sequence:
                sites = self.lattice.sites_of_species(initial)
                np.random.shuffle(sites)
                flip_list += [(s, final) for s in sites]
            num_active_sites = len(flip_list)
            w1 = w/float(num_active_sites)
            done_any = done_two = False
            for n, (site, species) in enumerate(flip_list):
                self.add_percolating_site(site=site, species=species)
                wrapping = self._is_wrapping[self.largest_cluster]
                if (np.sum(wrapping) > 0) and not done_any:
                    pc_site_any += w1*float(n+1)
                    pc_bond_any += w2*float(self._nbonds)
                    done_any = True
                    if file_name:
                        self.save_cluster(
                            self.largest_cluster,
                            file_name=(file_name+("-1d_%05d" % (i,))))
                    comp = self.lattice.composition
                    for s in comp:
                        if s in percolating_composition:
                            percolating_composition[s] += comp[s]*w
                        else:
                            percolating_composition[s] = comp[s]*w
                if (np.sum(
                        np.where(wrapping > 0, 1, 0)) >= 2) and not done_two:
                    pc_site_two += w1*float(n+1)
                    pc_bond_two += w2*float(self._nbonds)
                    done_two = True
                    if file_name:
                        self.save_cluster(
                            self.largest_cluster,
                            file_name=(file_name + ("-2d_%05d" % (i,))))
                if np.all(wrapping > 0):
                    pc_site_all += w1*float(n+1)
                    pc_bond_all += w2*float(self._nbonds)
                    if file_name:
                        self.save_cluster(
                            self.largest_cluster,
                            file_name=(file_name + ("-3d_%05d" % (i,))))
                    break
                if n == num_active_sites-1:
                    stderr.write(
                        "Error: All sites occupied, but no wrapping "
                        "cluster!?\n       "
                        "Maybe you defined a percolation rule that "
                        "never percolates.\n       "
                        "Have a look at `ERROR.vasp'.\n")
                    self.save_cluster(self.largest_cluster,
                                      file_name="ERROR.vasp")
                    sys.exit()

        pb()

        uprint(" Average percolating composition: ", end="")
        for s in percolating_composition:
            uprint("{} {:.2f} ".format(s, percolating_composition[s]), end="")
        uprint("\n")

        return (pc_site_any, pc_site_two, pc_site_all,
                pc_bond_any, pc_bond_two, pc_bond_all)

    def save_cluster(self, cluster, file_name="CLUSTER.vasp",
                     sort_species=True):
        """
        Save a particular cluster to an output file.
        Relies on `pymatgen' for the file I/O.

        Arguments:
          cluster (int or list): either the ID of a cluster or a list of
            all sites within a cluster
          file_name (str): name of the POSCAR file to be created
          sort_species (bool): if True, the coordinates will be sorted
            by species in the POSCAR file

        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vasp.inputs import Poscar

        species = ["V" for i in range(self._nsites)]
        if isinstance(cluster, (int, np.int64)):
            i = self._first[cluster]
            species[i] = "O"
            while (self._next[i] >= 0):
                i = self._next[i]
                species[i] = "O"
        else:
            for i in cluster:
                species[i] = "O"

        species = np.array(species)
        if sort_species:
            idx = np.argsort(species)
        else:
            idx = np.array([i for i in range(len(species))])

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

    def save_neighbors(self, site, file_name="NEIGHBORS.vasp"):
        """
        Save neighbors of site SITE to an output file.
        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vaspio import Poscar

        species = ["V" for i in range(self._nsites)]
        species[site] = "O"
        for nb in self.neighbors[site]:
            species[nb] = "O"

        species = np.array(species)
        idx = np.argsort(species)

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

    def compute_surface_area(self, avec):
        """
        Compute the surface area of the simulation cell defined by the
        lattice vector matrix AVEC.
        """

        A = 0.0
        A += 2.0*np.linalg.norm(np.cross(avec[0], avec[1]))
        A += 2.0*np.linalg.norm(np.cross(avec[0], avec[2]))
        A += 2.0*np.linalg.norm(np.cross(avec[1], avec[2]))
        return A

    def _count_surface_sites(self, dr=0.1):
        """
        Count the number of sites at the cell boundaries.
        This number is needed to determine the percolation flux.

        Arguments:
          dr   allowed fluctuation in the coordinates is +/- dr
        """

        nsurf = 0

        dr2 = 2.0*dr

        # first lattice direction
        idx = np.argsort(self._coo[:, 0])
        minval = self._coo[idx[0], 0]
        for i in idx:
            if (self._coo[i, 0] < minval + dr2):
                nsurf += 2
            else:
                break

        # second lattice direction
        idx = np.argsort(self._coo[:, 1])
        minval = self._coo[idx[0], 1]
        for i in idx:
            if (self._coo[i, 1] < minval + dr2):
                nsurf += 2
            else:
                break

        # third lattice direction
        idx = np.argsort(self._coo[:, 2])
        minval = self._coo[idx[0], 2]
        for i in idx:
            if (self._coo[i, 2] < minval + dr2):
                nsurf += 2
            else:
                break

        self._nsurface = nsurf

    def _merge_clusters(self, cluster1, site1, cluster2, site2, T2):
        """
        Add sites of cluster2 to cluster1.

        """

        if not self.check_bond(site1, site2):
            return

        # remember bonds
        nb1 = self.neighbors[site1].index(site2)
        if not self._bonds[site1][nb1]:
            nb2 = self.neighbors[site2].index(site1)
            self._bonds[site1][nb1] = True
            self._bonds[site2][nb2] = True
            self._nbonds += 1

        # vector from head node of cluster2 to head of cluster1
        v_12 = (self._coo[site2] + T2) - self._coo[site1]
        vec = self._vec[site1] - v_12 - self._vec[site2]

        if (cluster1 == cluster2):
            # if `vec' is different from the stored vector, we have
            # a wrapping cluster, i.e. we found the periodic image of
            # a site that is already part of the cluster
            wrapping1 = np.sum(self._is_wrapping[cluster1])
            if abs(vec[0]) > 0.5:
                self._is_wrapping[cluster1][0] += 1
                self._npaths += 1
            if abs(vec[1]) > 0.5:
                self._is_wrapping[cluster1][1] += 1
                self._npaths += 1
            if abs(vec[2]) > 0.5:
                self._is_wrapping[cluster1][2] += 1
                self._npaths += 1
            if (wrapping1 == 0) and np.sum(self._is_wrapping[cluster1]) > 0:
                self._npercolating += self._size[cluster1]
                for s, c in self._cluster_comp[cluster1].items():
                    if s in self._percol_comp:
                        self._percol_comp[s] += c
                    else:
                        self._percol_comp[s] = c
                self._nclus_percol += 1
            return
        else:
            # keep track of the number of sites in wrapping clusters
            wrapping1 = np.sum(self._is_wrapping[cluster1])
            wrapping2 = np.sum(self._is_wrapping[cluster2])
            if (wrapping1 > 0) and (wrapping2 == 0):
                self._npercolating += self._size[cluster2]
                for s, c in self._cluster_comp[cluster2].items():
                    if s in self._percol_comp:
                        self._percol_comp[s] += c
                    else:
                        self._percol_comp[s] = c
            elif (wrapping1 == 0) and (wrapping2 > 0):
                self._npercolating += self._size[cluster1]
                for s, c in self._cluster_comp[cluster1].items():
                    if s in self._percol_comp:
                        self._percol_comp[s] += c
                    else:
                        self._percol_comp[s] = c
            elif (wrapping1 > 0) and (wrapping2 > 0):
                self._nclus_percol -= 1

        # add vec to all elements of the second cluster
        # and change their cluster ID
        i = self._first[cluster2]
        self._vec[i, :] += vec
        self.cluster[i] = cluster1
        while (self._next[i] >= 0):
            i = self._next[i]
            self._vec[i, :] += vec
            self.cluster[i] = cluster1

        # insert second cluster right after the head node in
        # cluster 1
        j = self._first[cluster1]
        self._next[i] = self._next[j]
        self._next[j] = self._first[cluster2]

        # keep track of the cluster sizes and the largest cluster
        self._size[cluster1] += self._size[cluster2]
        if (self._size[cluster1] > self._size[self.largest_cluster]):
            self.largest_cluster = cluster1

        # keep track of the wrapping property
        l1 = self._is_wrapping[cluster1]
        l2 = self._is_wrapping[cluster2]
        self._is_wrapping[cluster1] = l1 + l2

        # combined cluster composition
        for s, c2 in self._cluster_comp[cluster2].items():
            if s in self._cluster_comp[cluster1]:
                self._cluster_comp[cluster1][s] += c2
            else:
                self._cluster_comp[cluster1][s] = c2

        # Only delete the cluster, if it is the last in the list.
        # Otherwise we would have to update the cluster IDs on all sites.
        self._nclusters -= 1
        if (len(self._first) == cluster2+1):
            del self._first[cluster2]
            del self._size[cluster2]
            del self._is_wrapping[cluster2]
        else:
            self._first[cluster2] = -1
            self._size[cluster2] = 0
            self._is_wrapping[cluster2] = [0, 0, 0]

    def check_bond(self, site1, site2):
        """
        Check, if the bond between two sites is percolating.

        Arguments:
          site1, site2 (int): Indices of the two sites

        Returns:
          True, if the bond is percolating
          False, otherwise

        """

        # check stability of both sites
        sl1 = self.lattice.site_labels[site1]
        sl2 = self.lattice.site_labels[site2]
        if sl1 in self.site_rules:
            for sr in self.site_rules[sl1]:
                if not sr(self, site1):
                    return False
        if sl2 in self.site_rules:
            for sr in self.site_rules[sl2]:
                if not sr(self, site2):
                    return False

        # check whether the bond is percolating
        bond = Bond(sl1, sl2)
        if bond in self.bond_rules:
            for br in self.bond_rules[bond]:
                if not br(self, site1, site2):
                    return False
        else:
            return False

        return True
