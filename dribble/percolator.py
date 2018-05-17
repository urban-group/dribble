# ----------------------------------------------------------------------
# This file is part of the 'Dribble' package for percolation simulations.
# Copyright (c) 2013-2018 Alexander Urban (aurban@atomistic.net)
# ----------------------------------------------------------------------
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Mozilla Public License, v. 2.0, for more details.

"""
Implementations of lattice percolation techniques.

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
    The Percolator class implements several techniques to evaluate
    percolation properties of a given structure or lattice.  This
    includes a fast MC algorithm for percolation analysis on regular
    lattices [1].  Several different methods allow the computation of
    various quantities related to percolation, such as the percolation
    susceptibility, the ratio of the largest cluster of sites to all
    occupied sites, the probability for wrapping (periodic) clusters
    vs. the site concentration, and the percolation threshold.

    Note on the terminology:

    Site percolation traditionally considers only two states for each
    site: `occupied` or `vacant`.  The terminology used in this class
    therefore refers to sites occupied by "percolating" species as
    "occupied", and all other sites are considered "vacant" irrespective
    of their species.  For example, in a lithium conductor all Li sites
    would be dubbed "occupied" whereas sites with any other species
    would be considered "vacant".  Often, vacancies are actually a
    percolating species, in which case vacancy sites would be considered
    "occupied" here!

    Methods for the analysis of static structures:

       check_spanning - Analyze percolating domains in a structure
       get_cluster_of_site - Determine connected sites
       extend_cluster - Grow percolating domains

    Methods for the (MC) calculation of percolation properties on
    regular lattices:

       calc_p_infinity - Calc. prob. of infinite cluster and susceptibility
       calc_p_wrapping - Calc. prob. of wrapping cluster
       inaccessible_sites - Calculate non-percolating fraction
       percolation_point - Calculate the percolation threshold

       The MC routines can draw progress bars to standard out to
       indicate the progress of the simulation.

    Auxiliary methods:

       save_clusters - Export a structure file for a select cluster
       save_neighbors - Export the neighbors of a select site

    Attributes:

       num_sites - Total number of sites
       occupied[i] - i-th occupied site (see terminology above)
       vacant[i] - i-th vacant site (see terminology above)
       num_occupied - Total number of occupied sites (see terminology above)
       num_vacant - Total number of vacant sites (see terminology above)
       num_percolating - Number of sites within percolating domains

       bond[i][j] - True, if site i is bound to its j-th neighbor site
       num_bonds - Number of bonds between sites
       num_bonds_max - Max. number of bonds possible for given lattice

       cluster[i] - ID of the domain (cluster) associated with site i;
           (cluster[i] < 0) if site i is vacant
       cluster_size[i] - Number of sites within cluster i
       largest_cluster - ID of the domain (cluster) with most sites
       num_clusters - Total number of domains (clusters) of occupied sites
       num_clus_percol - Number of percolating (periodically wrapping) domains
       wrapping[i][j] - Number of periodically wrapping paths within
           cluster i in lattice direction j (j = 0, 1, 2)
       num_paths - Total number of periodically wrapping paths found


       progress_bar_char - UTF-8 character used to draw progress bars

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

        self.lattice = lattice
        self.percolating_species = percolating_species
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
        self.cluster = self.lattice._occup
        self.occupied = self.lattice._occupied
        self.vacant = self.lattice._vacant

        self.bond = []
        # max. number of bonds is half the number of nearest neighbors
        self.num_bonds_max = 0
        for i in range(self._nsites):
            nbs = len(self.neighbors[i])
            if nbs > len(set(self.neighbors[i])):
                raise ValueError("Cell size too small. Try supercell.")
            self.num_bonds_max += nbs
            self.bond.append(np.array(nbs*[False]))
        self.num_bonds_max /= 2

        self.initially_occupied_sites = lattice._occupied[:]
        self.initial_species = [self.lattice.species[i]
                                for i in self.initially_occupied_sites]
        self.reset()

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

    def reset(self, occupied=None, species=None):
        """
        Reset the instance to the state of initialization.
        Static occupied sites are ensured to be always occupied.

        """

        if occupied is None:
            occupied = self.initially_occupied_sites[:]

        if species is None:
            species = self.initial_species[:]

        if not len(species) == len(occupied):
            raise ValueError("Error: the number of initial species has "
                             "to be identical to the number of occupied "
                             "sites.")

        self.num_clusters = 0
        self.num_bonds = 0
        self.num_percolating = 0
        self.num_clus_percol = 0
        self.num_paths = 0

        self.cluster_size = []
        self.wrapping = []

        # _first[i]       first site (head) of the i-th cluster
        # _next[i]        the next site in the same cluster as site i;
        #                 < 0, if site i is the final site
        self._first = []
        self._next = np.empty(self._nsites, dtype=int)
        self._next[:] = -1

        # _vec[i][j]      j-th component of the vector that connects
        #                 site i with the head site of the cluster
        self._vec = np.zeros(self._coo.shape)

        self.largest_cluster = -1

        # initial random species decoration, if specified
        # species = [None for i in occupied]
        if len(occupied) == 0 or self._initial_concentrations is not None:
            self.lattice.random_species_decoration(
                self._initial_concentrations,
                occupying_species=self.percolating_species,
                static=self._static_species)
            occupied = self.lattice._occupied[:]
            species = [self.lattice.species[i] for i in occupied]

        # completely reset lattice
        # _occupied[i]    i-th occupied site
        # _vacant[i]      i-th vacant site
        self.cluster[:] = -1
        self.occupied = []
        self.vacant = list(range(self._nsites))

        # Composition of all clusters and total composition of percolating
        # clusters
        self._cluster_comp = {}
        self._percol_comp = {}

        for i in range(self._nsites):
            self.bond[i][:] = False

        # repopulate initial occupations, if any:
        for i, s in enumerate(occupied):
            self._add_percolating_site(site=s, species=species[i])

    def __str__(self):
        ostr = "\n An instance of the Percolator class\n\n"
        ostr += " Structure info:\n"
        ostr += str(self.lattice)
        return ostr

    def __repr__(self):
        return self.__str__()

    @property
    def num_sites(self):
        return self.lattice.num_sites

    @property
    def num_sites_not_static(self):
        return self.lattice.num_sites_not_static

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

    @property
    def cluster_set(self):
        return set(self.cluster) - {-1}

    @property
    def percolating_clusters(self):
        """ Return set with IDs of all percolating clusters. """
        clusters = set()
        for cl in self.cluster_set:
            if np.sum(self.wrapping[cl]) > 0:
                clusters.add(cl)
        return clusters

    @property
    def percolating_sites(self):
        """ Return list of all sites that are part of percolating clusters """
        sites = []
        for cl in self.percolating_clusters:
            sites.extend(self.get_sites_of_cluster(cl))
        return sites

    def get_common_neighbors(self, site1, site2):
        """
        Returns a list of common neighbor sites of SITE1 and SITE2.
        """

        return [nb for nb in self.neighbors[site1]
                if nb in self.neighbors[site2]]

    def get_cluster_of_site(self, site, vec=[0, 0, 0], visited=[]):
        """
        Recursively determine all sites connected to a given site.

        Args:
          site (int): ID of the input site
          vec (list or array): Vector pointing from the head of the cluster
              to the site
          visited (list): List of sites that have already been visited
              by recursion

          `vec` and `visited` do not have to be specified manually but
          are only used in the recursive calls to this method.

        Returns:
          tuple (sites, nspanning) where
              sites (list): List of all sites connected with the input site
              nspanning (array): Counts of spanning paths in the three
                  lattice directions.

        """

        if (self.cluster[site] < 0):
            # vacant site (why are we here?)
            return visited

        nspanning = np.array([0, 0, 0], dtype=int)
        newsites = [site]

        for i in range(len(self.neighbors[site])):
            nb = self.neighbors[site][i]
            # neighboring site occupied and bound?
            if ((self.cluster[nb] >= 0) and self._check_bond(site, nb)):
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

    def get_sites_of_cluster(self, cluster):
        """
        Get all sites within a given cluster.

        Args:
          cluster (int): ID of the cluster

        Returns:
          sites (list)

        """

        cluster_sites = []
        site = self._first[cluster]
        while site >= 0:
            cluster_sites.append(site)
            site = self._next[site]
        return cluster_sites

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

    def extend_cluster(self, cluster_id, site_rules=None,
                       bond_rules=None, verbose=False):
        """
        Grow a single cluster of occupied sites by allowing additional bonds
        based on alternative site and/or bond rules.  The cluster is
        extended by one neighbor shell, so subsequent calls to this
        method may result in further growth.

        Args:
          cluster_id (int): ID of the cluster that is to be grown
          site_rules (dict): A set of alternative site rules
          bond_rules (dict): A set of alternative bond rules
          verbose (bool): Print operations to standard out

          Site and bond rules are dictionaries in the following format:

          site_rules = {Site('A'): [<Rule1>, [<Rule2>], ...]}
          bond_rules = {Bond('A', 'B'): [<Rule1>, [<Rule2>], ...]}

          Site and bond rules are implemented in `.rules.py`.

        """

        # original cluster ID
        cluster = cluster_id

        # all sites of the original cluster:
        cluster_sites = self.get_sites_of_cluster(cluster)

        # iterate over sites in cluster
        for site in cluster_sites:
            # check the neighbor sites of the cluster site.  There are three
            # different possibilities:
            # (1) The neighbor site is occupied and belongs to the same
            #     cluster;
            # (2) The neighbor site is occupied and belongs to a different
            #     cluster;
            # (3) The neighbor site is vacant.
            # Option (3) is irrelevant here, but the other two cases have
            # to be accounted for.  Option (1) is relevant only if no bond
            # between the two sites previously existed.
            for inb, nb in enumerate(self.neighbors[site]):
                nb_cl = self.cluster[nb]
                if (nb_cl >= 0) and not self.bond[site][inb]:
                    merged = self._merge_clusters(cluster, site, nb_cl, nb,
                                                  self._T_vectors[site][inb],
                                                  site_rules=site_rules,
                                                  bond_rules=bond_rules)
                    if verbose:
                        if merged and (cluster == nb_cl):
                            print("New bond between "
                                  "sites {} and {} in cluster {}.".format(
                                      site, nb, cluster))
                            if np.sum(self.wrapping[cluster]) > 0:
                                print(">>The cluster is wrapping.")
                        elif merged:
                            print("Clusters {} and {} merged.".format(
                                cluster, nb_cl))
                        else:
                            print("No bond formed between "
                                  "sites {} and {}.".format(site, nb))

    def extend_sites(self, site_rules=None, bond_rules=None,
                     verbose=False):
        """
        Allow each occupied site to form (at most) one additional bond based
        on alternative site and/or bond rules.  Subsequent calls to this
        method may result in further bonding.

        Args:
          site_rules (dict): A set of alternative site rules
          bond_rules (dict): A set of alternative bond rules
          verbose (bool): Print operations to standard out

          Site and bond rules are dictionaries in the following format:

          site_rules = {Site('A'): [<Rule1>, [<Rule2>], ...]}
          bond_rules = {Bond('A', 'B'): [<Rule1>, [<Rule2>], ...]}

          Site and bond rules are implemented in `.rules.py`.

        """

        # keep track of visited sites
        visited = []

        for site in self.occupied:
            if site in visited:
                continue
            cluster = self.cluster[site]
            # check the neighbor sites of the cluster site.  There are three
            # different possibilities:
            # (1) The neighbor site is occupied and belongs to the same
            #     cluster;
            # (2) The neighbor site is occupied and belongs to a different
            #     cluster;
            # (3) The neighbor site is vacant.
            # Option (3) is irrelevant here, but the other two cases have
            # to be accounted for.  Option (1) is relevant only if no bond
            # between the two sites previously existed.
            for inb, nb in enumerate(self.neighbors[site]):
                if (nb in visited) or (site in visited):
                    continue
                nb_cl = self.cluster[nb]
                if (nb_cl >= 0) and not self.bond[site][inb]:
                    merged = self._merge_clusters(cluster, site, nb_cl, nb,
                                                  self._T_vectors[site][inb],
                                                  site_rules=site_rules,
                                                  bond_rules=bond_rules)
                    if merged:
                        visited.append(site)
                        visited.append(nb)
                    if verbose:
                        if merged and (cluster == nb_cl):
                            print("New bond between "
                                  "sites {} and {} in cluster {}.".format(
                                      site, nb, cluster))
                            if np.sum(self.wrapping[cluster]) > 0:
                                print(">>The cluster is wrapping.")
                        elif merged:
                            print("Clusters {} and {} merged.".format(
                                cluster, nb_cl))
                        else:
                            print("No bond formed between "
                                  "sites {} and {}.".format(site, nb))

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
                self._add_percolating_site(site=site, species=species)
                Pn[n] += w*(float(self.cluster_size[self.largest_cluster]
                                  )/float(n+1))
                for cl in range(len(self.cluster_size)):
                    if cl == self.largest_cluster:
                        continue
                    Xn[n] += w2*self.cluster_size[cl]**2/float(n+1)
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
                self._add_percolating_site(site=site, species=species)
                wrapping = np.sum(self.wrapping[self.largest_cluster])
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
                self._add_percolating_site(site=site, species=species)
                N_ref = len(self.lattice.sites_of_species(species))
                try:
                    N_ref_percol = self._percol_comp[species]
                except KeyError:
                    N_ref_percol = 0
                Pn[n] += w*float(N_ref - N_ref_percol)/float(N_ref)
                # Pn[n] += w*float(n+1-self.num_percolating)/float(n+1)
                Qn[n] += w*float(self.num_clus_percol)/float(self.num_clusters)
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
        w2 = w/float(self.num_bonds_max)
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
                self._add_percolating_site(site=site, species=species)
                wrapping = self.wrapping[self.largest_cluster]
                if (np.sum(wrapping) > 0) and not done_any:
                    pc_site_any += w1*float(n+1)
                    pc_bond_any += w2*float(self.num_bonds)
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
                    pc_bond_two += w2*float(self.num_bonds)
                    done_two = True
                    if file_name:
                        self.save_cluster(
                            self.largest_cluster,
                            file_name=(file_name + ("-2d_%05d" % (i,))))
                if np.all(wrapping > 0):
                    pc_site_all += w1*float(n+1)
                    pc_bond_all += w2*float(self.num_bonds)
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

    def save_structure(self, file_name="percolating_sites.vasp",
                       sort_species=True, label="P"):
        """
        Save the current lattice decoration to an output file, labeling all
        sites that are connected to percolating clusters (if
        any). Relies on `pymatgen' for the file I/O.

        Arguments:
          file_name (str): name of the POSCAR file to be created
          sort_species (bool): if True, the coordinates will be sorted
            by species in the POSCAR file
          label (str): Species to be used to indicate percolating sites.

        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vasp.inputs import Poscar

        species = np.array(self.lattice.species[:])
        for site in self.percolating_sites:
            species[site] = label

        if sort_species:
            idx = np.argsort(species)
        else:
            idx = np.array([i for i in range(len(species))])

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

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

    def _add_percolating_site(self, site=None, species=None):
        """
        Change status of SITE to be percolating.

        Args:
          site    optional site index; if not specified, the site will be
                  randomly selected
          species optional species that will be assigned with the newly
                  percolating site

        Returns:
          merge_log (list): List of all merged clusters (if any) resulting
              from the additional site in the following format:
              [(c1, c2), (c3, c4), ...]
              Where the first cluster id in each tuple (c1, c3) gives the
              cluster that does no longer exist, and the second id (c2, c4)
              is the cluster that absorbed the other one.

        Note: The present implementation only considers bonds between
              nearest-neighbor sites!

        """

        if (self.num_vacant_not_static <= 0):
            stderr.write("Warning: all sites are already occupied\n")
            return

        if site is None:
            sel = np.random.random_integers(0, self.num_vacant_not_static-1)
            site = self.vacant_not_static[sel]
        elif (site in self.vacant) and (site not in self._static_vacant):
            sel = self.vacant.index(site)
        else:
            stderr.write("Warning: Attempt to occupy invalid site.\n")
            if site in self.occupied:
                stderr.write(
                    "         Site {} is already occupied.\n".format(site))
            if site in self._static_vacant:
                stderr.write(
                    "         Site {} is a static vacancy.\n".format(site))
            return

        del self.vacant[sel]
        self.occupied.append(site)
        self.lattice.species[site] = species

        # for the moment, add a new cluster
        self._first.append(site)
        self.cluster_size.append(1)
        self.wrapping.append(np.array([0, 0, 0]))
        cl = len(self._first) - 1
        self.cluster[site] = cl
        self._cluster_comp[cl] = {species: 1}
        self.num_clusters += 1
        self._vec[site, :] = [0.0, 0.0, 0.0]
        if (self.largest_cluster < 0):
            self.largest_cluster = self.cluster[site]

        # check, if this site
        # - defines a new cluster,
        # - will be added to an existing cluster, or
        # - connects multiple existing clusters.
        merge_log = []
        for i in range(len(self.neighbors[site])):
            nb = self.neighbors[site][i]
            cl = self.cluster[nb]
            if (cl >= 0):
                # only consider bonds between nearest neighbors
                if nb in self.lattice._nbshells[site][0]:
                    self._merge_clusters(cl, nb, self.cluster[site],
                                         site, -self._T_vectors[site][i])
                    if cl != self.cluster[site]:
                        merge_log.append((self.cluster[site], cl))

                # update also next nearest neighbors
                # loop over the neighbors of the neighbor
                for j in range(len(self.neighbors[nb])):
                    nb2 = self.neighbors[nb][j]
                    cl2 = self.cluster[nb2]
                    # also here: consider only neares-neighbor bonds
                    if (cl2 >= 0) and (nb2 in self.lattice._nbshells[nb][0]):
                        self._merge_clusters(cl2, nb2, self.cluster[nb],
                                             nb, -self._T_vectors[nb][j])
                        if cl2 != self.cluster[site]:
                            merge_log.append((self.cluster[site], cl2))

        return merge_log

    def _merge_clusters(self, cluster1, site1, cluster2, site2, T2,
                        site_rules=None, bond_rules=None):
        """
        Try to add sites of cluster2 to cluster1 by forming a bond between
        two sites.

        Args:
          cluster1 (int): ID of the first cluster
          site1 (int): Contact site of the first cluster
          cluster2 (int): ID of the second cluster
          site2 (int): Contact site of the second cluster
          T2 (array): vector connecting the two sites
          site_rules (dict): site rules to be used; if not given, use
              the default rules
          bond_rules (dict): bond rules to be used; if not given, use
              the default rules

          Site and bond rules are dictionaries in the following format:

          site_rules = {Site('A'): [<Rule1>, [<Rule2>], ...]}
          bond_rules = {Bond('A', 'B'): [<Rule1>, [<Rule2>], ...]}

          Site and bond rules are implemented in `.rules.py`.

        Returns:
          merged (bool): True if the bond has been created, otherwise False

        """

        if not self._check_bond(site1, site2, site_rules=site_rules,
                                bond_rules=bond_rules):
            return False

        # remember bonds
        nb1 = self.neighbors[site1].index(site2)
        if not self.bond[site1][nb1]:
            nb2 = self.neighbors[site2].index(site1)
            self.bond[site1][nb1] = True
            self.bond[site2][nb2] = True
            self.num_bonds += 1

        # vector from head node of cluster2 to head of cluster1
        v_12 = (self._coo[site2] + T2) - self._coo[site1]
        vec = self._vec[site1] - v_12 - self._vec[site2]

        if (cluster1 == cluster2):
            # if `vec' is different from the stored vector, we have
            # a wrapping cluster, i.e. we found the periodic image of
            # a site that is already part of the cluster
            wrapping1 = np.sum(self.wrapping[cluster1])
            if abs(vec[0]) > 0.5:
                self.wrapping[cluster1][0] += 1
                self.num_paths += 1
            if abs(vec[1]) > 0.5:
                self.wrapping[cluster1][1] += 1
                self.num_paths += 1
            if abs(vec[2]) > 0.5:
                self.wrapping[cluster1][2] += 1
                self.num_paths += 1
            if (wrapping1 == 0) and np.sum(self.wrapping[cluster1]) > 0:
                self.num_percolating += self.cluster_size[cluster1]
                for s, c in self._cluster_comp[cluster1].items():
                    if s in self._percol_comp:
                        self._percol_comp[s] += c
                    else:
                        self._percol_comp[s] = c
                self.num_clus_percol += 1
            return True
        else:
            # keep track of the number of sites in wrapping clusters
            wrapping1 = np.sum(self.wrapping[cluster1])
            wrapping2 = np.sum(self.wrapping[cluster2])
            if (wrapping1 > 0) and (wrapping2 == 0):
                self.num_percolating += self.cluster_size[cluster2]
                for s, c in self._cluster_comp[cluster2].items():
                    if s in self._percol_comp:
                        self._percol_comp[s] += c
                    else:
                        self._percol_comp[s] = c
            elif (wrapping1 == 0) and (wrapping2 > 0):
                self.num_percolating += self.cluster_size[cluster1]
                for s, c in self._cluster_comp[cluster1].items():
                    if s in self._percol_comp:
                        self._percol_comp[s] += c
                    else:
                        self._percol_comp[s] = c
            elif (wrapping1 > 0) and (wrapping2 > 0):
                self.num_clus_percol -= 1

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
        self.cluster_size[cluster1] += self.cluster_size[cluster2]
        if (self.cluster_size[cluster1] > self.cluster_size[self.largest_cluster]):
            self.largest_cluster = cluster1

        # keep track of the wrapping property
        l1 = self.wrapping[cluster1]
        l2 = self.wrapping[cluster2]
        self.wrapping[cluster1] = l1 + l2

        # combined cluster composition
        for s, c2 in self._cluster_comp[cluster2].items():
            if s in self._cluster_comp[cluster1]:
                self._cluster_comp[cluster1][s] += c2
            else:
                self._cluster_comp[cluster1][s] = c2

        # Only delete the cluster, if it is the last in the list.
        # Otherwise we would have to update the cluster IDs on all sites.
        self.num_clusters -= 1
        if (len(self._first) == cluster2+1):
            del self._first[cluster2]
            del self.cluster_size[cluster2]
            del self.wrapping[cluster2]
        else:
            self._first[cluster2] = -1
            self.cluster_size[cluster2] = 0
            self.wrapping[cluster2] = [0, 0, 0]

        return True

    def _check_bond(self, site1, site2, site_rules=None, bond_rules=None):
        """
        Check, if the bond between two sites is percolating.

        Arguments:
          site1, site2 (int): Indices of the two sites
          site_rules (dict): site rules to be used; if not given, use
              the default rules
          bond_rules (dict): bond rules to be used; if not given, use
              the default rules

          Site and bond rules are dictionaries in the following format:

          site_rules = {Site('A'): [<Rule1>, [<Rule2>], ...]}
          bond_rules = {Bond('A', 'B'): [<Rule1>, [<Rule2>], ...]}

          Site and bond rules are implemented in `.rules.py`.

        Returns:
          True, if the bond is percolating
          False, otherwise

        """

        sr = self.site_rules if site_rules is None else site_rules
        br = self.bond_rules if bond_rules is None else bond_rules

        # sublattices
        sl1 = self.lattice.site_labels[site1]
        sl2 = self.lattice.site_labels[site2]

        # check individual stability of both sites
        if sl1 in sr:
            for rule in sr[sl1]:
                if not rule(self, site1):
                    return False
        if sl2 in sr:
            for rule in sr[sl2]:
                if not rule(self, site2):
                    return False

        # check whether the bond between both sites is percolating
        bond = Bond(sl1, sl2)
        if bond in br:
            for rule in br[bond]:
                if not rule(self, site1, site2):
                    return False
        else:
            return False

        return True
