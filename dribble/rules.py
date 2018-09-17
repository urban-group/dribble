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
Site rules define whether a site can potentially be part of a
percolating network based on the occupancy of the surrounding sites.

Site rules should be implemented by extending the SiteRule class.

Bond rules define whether the bond between two sites is percolating,
which also usually depends on the neighboring sites.

Bond rules should be implemented by extending the BondRule class.

"""

from __future__ import print_function, division, unicode_literals
import abc
import numpy as np

__author__ = "Alexander Urban"
__email__ = "aurban@atomistic.net"
__date__ = "2017-07-24"
__version__ = "0.1"


class SiteRule(metaclass=abc.ABCMeta):
    """
    Abstract site stability rule class.  Extend this class to implement
    actual stability rules.

    Each stability rule has to implement the _check_stable method as
    templated here.

    Per default, a stability rule instance has one attribute:

      invert (bool): True if the rule should be inverted

    """

    def __init__(self, invert=False):
        self.invert = invert

    def __call__(self, percolator, site):
        return (self.invert != self._check_stable(percolator, site))

    def __str__(self):
        out = "Site Percolation Rule: {}".format(self.__class__.__name__)
        out += "\n    " + self.__doc__.strip() + "\n"
        return out

    @abc.abstractmethod
    def _check_stable(self, percolator, site):
        pass


class NeighborShellSR(SiteRule):
    """
    Site stability is determined by the neighbor shells.

    """

    def __init__(self, stable_nb_shells, invert=False):
        super(NeighborShellSR, self).__init__(invert=invert)
        self.stable_nb_shells = stable_nb_shells

    def _check_species(self, percolator, sites, rules):
        """
        Check if required min/max species counts are satisfied.
        """
        satisfied = True
        species = [percolator.lattice.species[s] for s in sites]
        for rule in rules:
            min_required = rule["min"] if "min" in rule else 0
            max_allowed = rule["max"] if "max" in rule else np.inf
            num_sites = len([s for s in species if s in rule["species"]])
            if (num_sites < min_required) or (num_sites > max_allowed):
                satisfied = False
                break
        return satisfied

    def _check_stable(self, percolator, site):
        stable = True
        for env in self.stable_nb_shells:
            # loop over neighbor shells in site environment
            for i, nbshell in enumerate(env):
                nb_list = percolator.lattice._nbshells[site][i]
                # loop over sublattices in current neighbor shell
                for sl in nbshell:
                    # determine neighbors on select sublattice
                    neighbors_on_sl = []
                    for s in nb_list:
                        if percolator.lattice.site_labels[s] == sl:
                            neighbors_on_sl.append(s)
                    stable &= self._check_species(
                        percolator, neighbors_on_sl, nbshell[sl])
        return stable


class BondRule(metaclass=abc.ABCMeta):
    """
    Abstract percolation rule class.  Extend this class to implement
    actual percolation rules.

    Each percolation rule has to implement the _check_percolating method
    as templated here.

    Per default, a percolation rule instance has oen attribute:

      invert (bool): True if the rule should be inverted

    """

    def __init__(self, invert=False):
        self.invert = invert

    def __call__(self, percolator, site1, site2):
        return (self.invert != self._check_percolating(
            percolator, site1, site2))

    def __str__(self):
        out = "Bond Percolation Rule: {}".format(self.__class__.__name__)
        out += "\n    " + self.__doc__.strip() + "\n"
        return out

    @abc.abstractmethod
    def _check_percolating(self, percolator, site1, site2):
        pass


class AllowedBondBR(BondRule):
    """
    True if the sublattices and species of the two sites are allowed to
    form bonds.

    """

    def __init__(self, sublattices=None, species=None, invert=False):
        """
        Arguments:
          sublattices (list): Allowed combination of sublattices as
              list of tuples or sets.
              Example:  Given three sublattices A, B, C, allowed_bonds
                        could be:  [{"A", "B"}, {"A", "C"}]
                        meaning that no direct percolation between "B"
                        and "C" is possible.
          species (list): Allowed combinations of species in the same
              format at the sublattices above.
          invert (bool): if True, the rule is inverted

        """
        super(AllowedBondBR, self).__init__(invert=invert)
        if sublattices is None:
            self.sublattices = None
        else:
            self.sublattices = [set(b) for b in sublattices]
        if species is not None:
            self.species = None
        else:
            self.species = [set(b) for b in species]

    def _check_percolating(self, percolator, site1, site2):
        percolating = True
        if self.sublattices is not None:
            sl1 = percolator.lattice.site_labels[site1]
            sl2 = percolator.lattice.site_labels[site2]
            percolating &= ({sl1, sl2} in self.sublattices)
        if self.species is not None:
            sl1 = percolator.lattice.species[site1]
            sl2 = percolator.lattice.species[site2]
            percolating &= ({sl1, sl2} in self.species)
        return percolating


class NearestNeighborBR(BondRule):
    """
    This rule is True when two sites are nearest neighbors.

    """

    def _check_percolating(self, percolator, site1, site2):
        return (site2 in percolator.neighbors[site1])


class CommonNeighborsBR(BondRule):
    """
    This rule is True when two sites have at exactly a specified number
    of common neighbors that are percolating.

    """

    def __init__(self, num_neighbors, invert=False):
        """
        Arguments:
          num_neighbors (int): Required number of common neighbors
          invert (bool): if True, the rule is inverted

        """
        super(CommonNeighborsBR, self).__init__(invert=invert)
        self.num_neighbors = num_neighbors

    def _check_percolating(self, percolator, site1, site2):
        common = [nb for nb in percolator._neighbors[site1]
                  if nb in percolator._neighbors[site1]]
        return (len(common) == self.num_neighbors)


class MinCommonNeighborsBR(BondRule):
    """
    This rule is True when two sites have at least a specified number of
    common neighbors that are percolating.

    Invert this rule to obtain a rule that limits the maximum number of
    common neighbors.

    """

    def __init__(self, num_neighbors, invert=False):
        """
        Arguments:
          num_neighbors (int): Required number of common neighbors
          invert (bool): if True, the rule is inverted

        """
        super(MinCommonNeighborsBR, self).__init__(invert=invert)
        self.num_neighbors = num_neighbors

    def _check_percolating(self, percolator, site1, site2):
        common_nb = percolator.get_common_neighbors(site1, site2)
        occupied = [nb for nb in common_nb
                    if percolator.cluster[nb] >= 0]
        return (len(occupied) >= self.num_neighbors)


class MinCommonNNNeighborsBR(BondRule):
    """
    This rule is True when two sites have at least a specified number of
    common neighbors that are percolating, and those common neighbors
    are themselves nearest neighbors.

    """

    def __init__(self, num_neighbors, invert=False):
        """
        Arguments:
          num_neighbors (int): Required number of common neighbors
          invert (bool): if True, the rule is inverted

        """
        super(MinCommonNNNeighborsBR, self).__init__(invert=invert)
        self.num_neighbors = num_neighbors

    def _check_percolating(self, percolator, site1, site2):
        common_nb = percolator.get_common_neighbors(site1, site2)
        occupied = [nb for nb in common_nb
                    if percolator.cluster[nb] >= 0]
        max_nn = 0
        for i, nb1 in enumerate(occupied):
            nn = 1
            for nb2 in occupied[i+1:]:
                if nb2 in percolator.neighbors[nb1]:
                    nn += 1
            max_nn = max(max_nn, nn)
        return (max_nn >= self.num_neighbors)
