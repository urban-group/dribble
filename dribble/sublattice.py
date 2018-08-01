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
Object class to hold information about sub-lattices.

"""

from __future__ import print_function, division

from . import rules

__author__ = "Alexander Urban"
__email__ = "aurban@atomistic.net"
__date__ = "2017-08-28"
__version__ = "0.1"


class Bond(object):
    """
    Bonds between sites of two sublattices.  The two sublattices can be
    identical.  Bond objects are hashable so that they can be used as
    dictionary keys.  Bonds are not directional, i.e., always
    Bond(A, B) == Bond(B, A).

    Attributes:
      sublattices (set): Set of the sublattice A and B
      bond_rules (list): List of bond rules (see rules.py)
      interactions (dict): Species-dependent interaction energies as
          interactions[{"A", "B"}]

    Methods:
      energy(A, B): Returns the interaction energy for species A and B
          if available and 0 otherwise; the order of A and B does not matter

    """

    def __init__(self, sublattice1, sublattice2, bond_rules=None,
                 interactions=None):
        """
        Arguments:
          sublattices (set or list): labels of the sublattices that the
            sites connected by the bond belong to.
          bond_rules (list): list of bond rules (see rules.py) that
            have to be fulfilled for this bond to be percolating
          interactions (list): interaction energies for pairs of species
            and the present bond; each interaction is a dict of form:
            {"species": ["A", "B"], "energy": E}

        """

        self.sublattices = frozenset([sublattice1, sublattice2])
        self.bond_rules = []
        self.interactions = {}

        if bond_rules is not None:
            for br in bond_rules:
                rule = getattr(rules, br[0])
                args = br[1] if len(br) > 1 else {}
                self.bond_rules.append(rule(**args))

        if interactions is not None:
            for ia in interactions:
                self.interactions[frozenset(ia["species"])] = ia["energy"]

    def __eq__(self, other):
        if hasattr(other, 'sublattices'):
            return self.sublattices == other.sublattices
        else:
            return self.sublattices == other

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if len(self.sublattices) == 2:
            s1, s2 = sorted(self.sublattices)
        else:
            s1 = s2 = list(self.sublattices)[0]
        return "Bond('{}', '{}')".format(s1, s2)

    def __hash__(self):
        return hash(self.sublattices)

    @classmethod
    def from_dict(cls, bond_dict):
        """
        Valid keys are 'sublattices' and 'bond_rules'.

        """
        sublattice1 = bond_dict["sublattices"][0]
        sublattice2 = bond_dict["sublattices"][1]
        if "bond_rules" in bond_dict:
            bond_rules = bond_dict["bond_rules"]
        else:
            bond_rules = None
        if "interactions" in bond_dict:
            interactions = bond_dict["interactions"]
        else:
            interactions = None
        return cls(sublattice1, sublattice2, bond_rules=bond_rules,
                   interactions=interactions)

    def energy(self, A, B):
        """
        Interaction energy for the present bond and species A and B.

        Args:
          A, B (str): The bond-forming species

        Returns:
          energy (float): If an interaction energy for the two species is
              defined it will be returned.  Otherwise, 0 is returned.

        """

        AB = frozenset([A, B])
        if AB in self.interactions:
            energy = self.interactions[AB]
        else:
            energy = 0.0
        return energy


class Sublattice(object):

    def __init__(self, structure, sites, description=None, species=None,
                 initial_occupancy=None, ignore=False, site_rules=None):
        """
        Arguments:
          structure (pymatgen Structure): the structure that this sublattice
            belongs to.
          sites: sites belonging to the sublattice; this can be either a
            list of site indices starting with 1 or a dictionary with
            'selectors'.  Currently, valid selectors are: "species"

                 Examples:
                   sites = [3, 4, 5, 6]
                   sites = {"species": ["Li", "Co"]}

          description (str): an optional short string describing the
            sublattice
          species (list): an optional list of species for each site
          initial_occupancy: an optional dictionary with relative
            concentrations
          ignore (bool): if True, ignore this sublattice in percolation
            simulations
          site_rules (list): a list of site rules (see rules.py)
            that have to be fulfilled for a site of the present sublattice
            can to become part of the percolating network.

        """
        self.structure = structure
        self.description = description
        self.ignore = ignore
        self.allowed_species = None

        self.site_rules = []
        if site_rules is not None:
            for sr in site_rules:
                rule = getattr(rules, sr[0])
                args = sr[1] if len(sr) > 1 else {}
                self.site_rules.append(rule(**args))

        try:
            self.sites = [int(s)-1 for s in sites]
        except ValueError:
            if "species" in sites:
                self.allowed_species = set(sites["species"])
                self.sites = [i for i, s in enumerate(self.structure)
                              if s.specie.symbol in sites["species"]]

        if species is None:
            # take species from structure
            self.species = [self.structure[i].specie.symbol
                            for i in self.sites]
        else:
            # species specified separately
            if len(species) != self.num_sites:
                raise ValueError("Number of species != number of sites.")
            self.species = species

        if self.allowed_species is None:
            self.allowed_species = set(self.species)
            if initial_occupancy is not None:
                self.allowed_species |= set(initial_occupancy.keys())

        self.initial_occupancy = initial_occupancy
        if self.initial_occupancy is not None:
            # make sure that concentrations add up to 1.0
            c_sum = sum([abs(c) for c in self.initial_occupancy.values()])
            if abs(1.0 - c_sum) > 0.001:
                raise ValueError(
                    "Concentrations do not add up to 1.0: "
                    + str(initial_occupancy))

    @property
    def num_sites(self):
        return len(self.sites)

    def __str__(self):
        out = "Sublattice:"
        if self.description is not None:
            out += "\n  Description: {}".format(self.description)
        if self.initial_occupancy is not None:
            out += "\n  Initial occupancy: " + ", ".join(
                ["{}: {}".format(s, self.initial_occupancy[s])
                 for s in self.initial_occupancy])
        out += "\n  Sites: " + ", ".join(str(s) for s in self.sites)
        if self.ignore:
            out += "\n  This sublattice is ignored in percolation simulations."
        return out
