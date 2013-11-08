""" Copyright (c) 2013 Alexander Urban

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

from __future__ import print_function

__autor__ = "Alexander Urban"
__date__  = "2013-02-15"

import numpy as np
import sys

from sys         import stderr
from scipy.stats import binom

from aux         import uprint
from aux         import ProgressBar
from lattice     import Lattice

EPS   = 100.0*np.finfo(np.float).eps

#----------------------------------------------------------------------#
#                           Percolator class                           #
#----------------------------------------------------------------------#

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

    def __init__(self, lattice):
        """
        Arguments:
          lattice  an instance of the Lattice class
        """

        """                    static data

        _special      True, if special percolation rules have been defined
        _bonds[i][j]        True, if there is a bond between the j-th site
                            and its j-th neighbor
        _nsurface           number of sites at cell boundary
                            (computed, if needed)
        _T_vectors[i][j]    the translation vector belonging to
                            _neighbors[i][j]
        _nbonds_tot         maximum number of possible bonds between the sites
        _cluster[i]     ID of cluster that site i belongs to; < 0, if
                        site i is vacant
        """

        self._lattice   = lattice
        # references for convenient access (no copies)
        self._avec      = self._lattice._avec
        self._coo       = self._lattice._coo
        self._nsites    = self._lattice._nsites
        self._neighbors = self._lattice._nn
        self._T_vectors = self._lattice._T_nn
        self._nsurface  = self._lattice._nsurface
        self._cluster   = self._lattice._occup
        self._occupied  = self._lattice._occupied
        self._vacant    = self._lattice._vacant

        self._special   = False

        self._bonds     = []
        # max. number of bonds is half the number of nearest neighbors
        self._nbonds_tot = 0
        for i in xrange(self._nsites):
            nbs = len(set(self._neighbors[i]))
            self._nbonds_tot += nbs
            self._bonds.append(np.array(nbs*[False]))
        self._nbonds_tot /= 2

        self.reset(init=True)

        sys.setrecursionlimit(50000)

    @classmethod
    def from_structure(cls, structure, **kwargs):
        """
        Create a Percolator instance based on the lattice vectors
        defined in a `structure' object.

        Arguments:
          structure       an instance of pymatgen.core.structure.Structure
          all keyword arguments are passed to the Lattice class constructor
        """

        avec    = structure.lattice.matrix
        coo     = structure.frac_coords
        lattice = Lattice(avec, coo, **kwargs)
        percol  = cls(lattice)

        return percol

    @classmethod
    def from_coordinates(cls, lattice_vectors, frac_coords, **kwargs):
        """
        Create a Percolator instance based on the lattice vectors
        defined in a `structure' object.

        Arguments:
          structure       an instance of pymatgen.core.structure.Structure
          all keyword arguments of the main constructor
        """

        lattice = Lattice(lattice_vectors, frac_coords)
        percol  = cls(lattice, **kwargs)

        return percol

    def reset(self, init=False, occupied=[]):
        """
        Reset the instance to the state of initialization.
        """

        """              internal dynamic data

        _cluster[i]     ID of cluster that site i belongs to; < 0, if
                        site i is vacant
        _nclusters      total number of clusters with more than 0 sites
                        Note: len(_first) can be larger than this !
        _nbonds         the number of bonds found so far
        _npercolating   number of sites that are members of percolating clusters
        _nclus_percol   number of percolating clusters
        _npaths         total number of percolation paths
        _first[i]       first site (head) of the i-th cluster
        _size[i]        size of cluster i (i.e., number of sites in i)
        _is_wrapping[i][j]  number of times that cluster i is wrapping in direction j
        _next[i]        the next site in the same cluster as site i;
                        < 0, if site i is the final site
        _vec[i][j]      j-th component of the vector that connects
                        site i with the head site of the cluster

        _occupied[i]    i-th occupied site
        _vacant[i]      i-th vacant site

        """

        self._nclusters    = 0
        self._nbonds       = 0
        self._npercolating = 0
        self._nclus_percol = 0
        self._npaths       = 0
        self._first        = []
        self._size         = []
        self._is_wrapping  = []
        self._largest      = -1
        self._next         = np.empty(self._nsites, dtype=int)
        self._next[:]      = -1
        self._vec          = np.zeros(self._coo.shape)

        if not init:
            self._cluster[:]   = -1
            self._occupied     = []
            self._vacant       = range(self._nsites)

        for i in range(self._nsites):
            self._bonds[i][:] = False

        # initial occupations, if any:
        for s in occupied:
            self.add_percolating_site(site=s)

    def __str__(self):
        ostr  = "\n An instance of the Percolator class\n\n"
        ostr += " Structure info:\n"
        ostr += str(self._lattice)
        return ostr

    def __repr__(self):
        return self.__str__()


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
    def num_vacant(self):
        return len(self._vacant)

    @property
    def num_occupied(self):
        return len(self._occupied)

    @property
    def num_sites(self):
        return self._nsites


    def get_cluster_of_site(self, site, vec=[0,0,0], visited=[]):
        """
        Recursively determine all other sites connected to SITE.
        """

        if (self._cluster[site] < 0):
            # vacant site (why are we here?)
            return visited

        nspanning = np.array([0,0,0], dtype=int)
        newsites  = [site]

        for i in xrange(len(self._neighbors[site])):
            nb = self._neighbors[site][i]
            # neighboring site occupied and bound?
            if ((self._cluster[nb] >= 0) and self._check_special(site,nb)):
                # vector to origin
                T = self._T_vectors[site][i]
                v_12 = (self._coo[nb] + T) - self._coo[site]
                vec_nb = vec + v_12
                if not (nb in visited+newsites):
                    self._vec[nb] = vec_nb
                    (subcluster, subspan
                    ) = self.get_cluster_of_site(nb, vec_nb, visited+newsites)
                    newsites  += subcluster
                    nspanning += subspan
                else:
                    # nb has already been visited,
                    # was it a different image?
                    diff = np.abs(self._vec[nb] - vec_nb)
                    nspanning += np.where(diff > 0.5, 1, 0)

        return (newsites, nspanning)

    def check_spanning(self, verbose=False):
        """
        Check, how many (if any) spanning clusters are present.
        """

        visited = []
        nspanning = 0

        for i in xrange(self._nsites):
            if (self._cluster[i] > 0) and not (i in visited):
                (cluster, spanning) = self.get_cluster_of_site(i)
                visited += cluster
                if np.sum(spanning) > 0:
                    nspanning += len(cluster)
                    if verbose:
                        uprint(" cluster with {} sites, ".format(len(cluster))
                               + "paths in (x,y,z): "
                               + "{} {} {}".format(*spanning))

        return nspanning


    def get_common_neighbors(self, site1, site2):
        """
        Returns a list of common neighbor sites of SITE1 and SITE2.
        """

        nb1 = self._neighbors[site1]
        nb2 = self._neighbors[site2]
        common = []
        for n in nb1:
            if n in nb2:
                common.append(n)
        return common

        # Note, the above is slightly faster than simply:
        # return list(set(self._neighbors[site1])
        #             & set(self._neighbors[site2]))


    def set_special_percolation_rule(self, num_common=0, same=None, require_NN=False, dr=0.1):
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
          require_NN   if True, the 'num_common' criterion additionally requires
                       the common nearest neighbors themselves to be
                       nearest neighbors (useful for diffusion on an fcc lattice)
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
            same_rule   = True
            NN_rule     = True

            if same and (0 <= same <= 2):
                if (abs(self._coo[site1][same]-self._coo[site2][same])<=dr):
                    same_rule = True
                else:
                    same_rule = False

            if (num_common > 0) and same_rule:
                common_rule = False
                common_nb = self.get_common_neighbors(site1, site2)
                occupied = []
                noccup   = 0
                for nb in common_nb:
                    if (self._cluster[nb] >= 0):
                        occupied.append(nb)
                        noccup += 1
                    if (noccup >= num_common):
                        common_rule = True

            if (require_NN and common_rule and same_rule):
                NN_rule = False
                for i in range(noccup):
                    site1 = occupied[i]
                    nb1 = self._neighbors[site1]
                    num_NN = 1
                    for j in range(i+1,noccup):
                        site2 = occupied[j]
                        if site2 in nb1:
                            num_NN +=1
                            if (num_NN >= num_common):
                                NN_rule = True
                                break

            percolating = (common_rule and same_rule and NN_rule)
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

        if (self.num_vacant <= 0):
            stderr.write("Warning: all sites are already occupied\n")
            return

        if site is None:
            sel = np.random.random_integers(0,self.num_vacant-1)
            site = self._vacant[sel]
        else:
            sel = self._vacant.index(site)

        del self._vacant[sel]
        self._occupied.append(site)

        # for the moment, add a new cluster
        self._first.append(site)
        self._size.append(1)
        self._is_wrapping.append(np.array([0, 0, 0]))
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

    def calc_p_infinity(self, plist, samples=500, save_discrete=False,
                        initial_occupations=False):
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

        # remember initial occupations, if desired
        if initial_occupations:
            occup0 = self._occupied[:]
        else:
            occup0 = []
        nocc = len(occup0)

        Pn = np.zeros(self._nsites)
        Xn = np.zeros(self._nsites)
        w  = 1.0/float(samples)
        w2 = w*float(self._nsites)
        for i in xrange(samples):
            pb()
            self.reset()
            np.random.shuffle(occup0)
            for n in xrange(self._nsites):
                if n < nocc:
                    self.add_percolating_site(site=occup0[n])
                else:
                    self.add_percolating_site()
                Pn[n] += w*(float(self._size[self._largest])/float(n+1))
                for cl in xrange(len(self._size)):
                    if cl == self._largest:
                        continue
                    Xn[n] += w2*self._size[cl]**2/float(n+1)

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

    def calc_p_wrapping(self, plist, samples=500, save_discrete=False,
                        initial_occupations=False):
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

        # remember initial occupations, if desired
        if initial_occupations:
            occup0 = self._occupied[:]
        else:
            occup0 = []
        nocc = len(occup0)

        Pn = np.zeros(self._nsites)
        Pnc = np.zeros(self._nsites)
        w  = 1.0/float(samples)
        w2 = w*(self._nsites)
        for i in xrange(samples):
            pb()
            self.reset()
            np.random.shuffle(occup0)
            for n in xrange(self._nsites):
                if n < nocc:
                    self.add_percolating_site(site=occup0[n])
                else:
                    self.add_percolating_site()
                wrapping = np.sum(self._is_wrapping[self._largest])
                if (wrapping > 0):
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

    def percolating_bonds(self, plist, samples=500, save_discrete=False,
                          initial_occupations=False):
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

        # remember initial occupations, if desired
        if initial_occupations:
            occup0 = self._occupied[:]
        else:
            occup0 = []
        nocc = len(occup0)

        Pn = np.zeros(self._nsites)
        w  = 1.0/float(samples)/float(self._nbonds_tot)
        for i in xrange(samples):
            pb()
            self.reset()
            np.random.shuffle(occup0)
            for n in xrange(self._nsites):
                if n < nocc:
                    self.add_percolating_site(site=occup0[n])
                else:
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

    def inaccessible_sites(self, plist, samples=500, save_discrete=False,
                           initial_occupations=False):
        """
        Estimate the number of inaccessible sites, i.e. sites that are
        not part of a percolating cluster, for a given range of
        concentrations.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete-wrap.dat)

        Returns:
          list of values corresponding to probabilities in `plist'
        """

        uprint(" Calculating fraction of inaccessible sites.")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        # remember initial occupations, if desired
        if initial_occupations:
            occup0 = self._occupied[:]
        else:
            occup0 = []
        nocc = len(occup0)

        Pn = np.zeros(self._nsites)
        Qn = np.zeros(self._nsites)
        w  = 1.0/float(samples)
        for i in xrange(samples):
            pb()
            self.reset()
            np.random.shuffle(occup0)
            for n in xrange(self._nsites):
                if n < nocc:
                    self.add_percolating_site(site=occup0[n])
                else:
                    self.add_percolating_site()
                Pn[n] += w*float(n+1-self._npercolating)/float(n+1)
                Qn[n] += w*float(self._nclus_percol)/float(self._nclusters)

        pb()

        if save_discrete:
            fname = 'discrete-inaccessible.dat'
            uprint("Saving discrete data to file: {}".format(fname))
            with open(fname, "w") as f:
                for n in xrange(self._nsites):
                    f.write("{} {} {}\n".format(n+1, Pn[n], Qn[n]))

        uprint(" Return convolution with a binomial distribution.\n")

        nlist = np.arange(1, self._nsites+1, dtype=int)
        Pp = np.empty(len(plist))
        Qp = np.empty(len(plist))
        for i in xrange(len(plist)):
            Pp[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Pn)
            Qp[i] = np.sum(binom.pmf(nlist, self._nsites, plist[i])*Qn)

        return (Pp, Qp)

    def percolation_flux(self, plist, samples=500, save_discrete=False,
                           initial_occupations=False):
        """
        Estimate the ratio of percolation pathes over all possible
        cell boundary sites.

        Arguments:
          plist          list with desired probability points p; 0 < p < 1
          samples        number of samples to average the MC result over
          save_discrete  if True, save the discrete, supercell dependent
                         values as well (file: discrete-wrap.dat)

        Returns:
          list of values corresponding to probabilities in `plist'
        """

        uprint(" Calculating percolation flux.")
        uprint(" Averaging over {} samples:\n".format(samples))

        A = self.compute_surface_area(self._avec)

        pb = ProgressBar(samples)

        # remember initial occupations, if desired
        if initial_occupations:
            occup0 = self._occupied[:]
        else:
            occup0 = []
        nocc = len(occup0)

        Pn = np.zeros(self._nsites)
        w  = 1.0/float(samples)/A
        for i in xrange(samples):
            pb()
            self.reset()
            np.random.shuffle(occup0)
            for n in xrange(self._nsites):
                if n < nocc:
                    self.add_percolating_site(site=occup0[n])
                else:
                    self.add_percolating_site()
                Pn[n] += w*self._npaths

        pb()

        if save_discrete:
            fname = 'discrete-flux.dat'
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

    def percolation_point(self, samples=500, file_name=None,
                          initial_occupations=False):
        """
        Determine an estimate for the site percolation threshold p_c.
        """

        uprint(" Calculating an estimate for the percolation point p_c.")
        uprint(" Averaging over {} samples:\n".format(samples))

        pb = ProgressBar(samples)

        # remember initial occupations, if desired
        if initial_occupations:
            occup0 = self._occupied[:]
        else:
            occup0 = []
        nocc = len(occup0)

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
            np.random.shuffle(occup0)
            done_any = done_two = False
            for n in xrange(self._nsites):
                if n < nocc:
                    self.add_percolating_site(site=occup0[n])
                else:
                    self.add_percolating_site()
                wrapping = self._is_wrapping[self._largest]
                if (np.sum(wrapping) > 0) and not done_any:
                    pc_site_any += w1*float(n+1)
                    pc_bond_any += w2*float(self._nbonds)
                    done_any = True
                    if file_name:
                        self.save_cluster(self._largest,
                             file_name=(file_name+("-1.%05d"%(i,))))
                if (np.sum(np.where(wrapping>0,1,0))>=2) and not done_two:
                    pc_site_two += w1*float(n+1)
                    pc_bond_two += w2*float(self._nbonds)
                    done_two = True
                    if file_name:
                        self.save_cluster(self._largest,
                             file_name=(file_name+("-2.%05d"%(i,))))
                if np.all(wrapping>0):
                    pc_site_all += w1*float(n+1)
                    pc_bond_all += w2*float(self._nbonds)
                    if file_name:
                        self.save_cluster(self._largest,
                             file_name=(file_name+("-3.%05d"%(i,))))
                    break
                if n == self._nsites-1:
                    stderr.write(
                        "Error: All sites occupied, but no wrapping cluster!?"
                        + "\n       "
                        + "Maybe you defined a percolation rule that never percolates."
                        + "\n       Have a look at `ERROR.vasp'.\n")
                    self.save_cluster(self._largest, file_name="ERROR.vasp")
                    print(wrapping)
                    sys.exit()

        pb()

        return (pc_site_any, pc_site_two, pc_site_all,
                pc_bond_any, pc_bond_two, pc_bond_all)


    def save_cluster(self, cluster, file_name="CLUSTER.vasp"):
        """
        Save a particular cluster to an output file.
        Relies on `pymatgen' for the file I/O.
        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vaspio      import Poscar

        species = ["V" for i in range(self._nsites)]
        i = self._first[cluster]
        species[i] = "O"
        while (self._next[i] >= 0):
            i = self._next[i]
            species[i] = "O"

        species = np.array(species)
        idx = np.argsort(species)

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

    def save_neighbors(self, site, file_name="NEIGHBORS.vasp"):
        """
        Save neighbors of site SITE to an output file.
        """

        from pymatgen.core.structure import Structure
        from pymatgen.io.vaspio      import Poscar

        species = ["V" for i in range(self._nsites)]
        species[site] = "O"
        for nb in self._neighbors[site]:
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


    #------------------------------------------------------------------#
    #                         private methods                          #
    #------------------------------------------------------------------#

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
        idx    = np.argsort(self._coo[:,0])
        minval = self._coo[idx[0],0]
        for i in idx:
            if (self._coo[i,0] < minval + dr2):
                nsurf += 2
            else:
                break

        # second lattice direction
        idx    = np.argsort(self._coo[:,1])
        minval = self._coo[idx[0],1]
        for i in idx:
            if (self._coo[i,1] < minval + dr2):
                nsurf += 2
            else:
                break

        # third lattice direction
        idx    = np.argsort(self._coo[:,2])
        minval = self._coo[idx[0],2]
        for i in idx:
            if (self._coo[i,2] < minval + dr2):
                nsurf += 2
            else:
                break

        self._nsurface = nsurf

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
                self._nclus_percol += 1
            return
        else:
            # keep track of the number of sites in wrapping clusters
            wrapping1 = np.sum(self._is_wrapping[cluster1])
            wrapping2 = np.sum(self._is_wrapping[cluster2])
            if (wrapping1 > 0) and (wrapping2 == 0):
                self._npercolating += self._size[cluster2]
            elif (wrapping1 == 0) and (wrapping2 > 0):
                self._npercolating += self._size[cluster1]
            elif (wrapping1 > 0) and (wrapping2 > 0):
                self._nclus_percol -= 1

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

        # keep track of the wrapping property
        l1 = self._is_wrapping[cluster1]
        l2 = self._is_wrapping[cluster2]
        self._is_wrapping[cluster1] = l1 + l2

        # Only delete the cluster, if it is the last in the list.
        # Otherwise we would have to update the cluster IDs on all sites.
        self._nclusters -= 1
        if (len(self._first) == cluster2+1):
            del self._first[cluster2]
            del self._size[cluster2]
            del self._is_wrapping[cluster2]
        else:
            self._first[cluster2] = -1
            self._size[cluster2]  = 0
            self._is_wrapping[cluster2] = [0, 0, 0]

    def _check_special(self, site1, site2):
        """
        Check, if the special percolation rule is fulfilled
        between sites SITE1 and SITE2.

        However, this instance does not define any special rule.
        """

        # This method may be replaced at run-time by calling
        # set_special_percolation_rule().

        return True
