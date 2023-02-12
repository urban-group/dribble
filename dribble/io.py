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
Parse JSON input files with percolation rules, site labels, etc.

"""

from __future__ import print_function, division, unicode_literals

import json
from warnings import warn
import numpy as np

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Structure

from .sublattice import Sublattice, Bond

__author__ = "Alexander Urban"
__email__ = "aurban@lbl.gov"
__date__ = "2017-02-14"
__version__ = "0.1"


class Input(object):
    """
    Parse JSON input file with specs for percolation simulation:

    Main keys:

      structure (str): Path to a POSCAR file with site coordinates
      formula_units (int): Number of formula units in the reference POSCAR
      cutoff (float): Distance cutoff for the neighbor list
      sublattices (dict): Sublattices (see below)
      bonds (list): Allowed bonds between sites of sublattices (see below);
          for example, if percolating channels between sites of sublattice
          "A" and "B" exist, a bond ["A", "B"] has to be defined here
      percolating_species (list): List of chemical species (str) that are
          percolating; this could be "Li" in Li-ion conductors
      static_species (list): List of chemical species (str) that are inactive,
          i.e., that never change during the simulation
      initial_occupancy (dict): Initial occupancies for each sublattice
          as dictionaries; for the example of two sublattices "O" and "T",
          a possible initial occupancy could be:
          {"O": {"Li":0.9, "Vac": 0.1}, "T": {"Vac": 1.0}}
      flip_sequence (list): List of pairs of chemical species ["A", "B"]
          that defines the order in which lattice sites are changed during
          the simulation

    Sublattices:

      Each sublattice has a unique label and is defined by a distionary
      with the following keys:

      description (str): A brief informative description of the sublattice
      sites (list): List of sites (int starting with 1) in the reference
          POSCAR that belong to the sublattice
      stable_neighbor_shells (dict): Site stability criterions in terms of
          coordination shells; for example, for two sublattices "T" and "O",
          and chemical species "Li" and "Vac" the following could be a
          meaningful stability criterion for sublattice "T":

            [
              {"O": [{"min": 3, "species": ["Vac"]}]},
              {"T": [{"max": 1, "species": ["Li"]}]}
            ]

          which expresses the following conditions for the first two
          neighbor shells:
          At least 3 of the "O" sites in the first neighbor shell have to
          be occupied by the "Vac" species.  At most 1 "T" site in the
          second neighbor shell may be occupied by "Li".

          Several alternative sets of conditions can be provided for each
          sublattice, i.e., 'stable_neighbor_shells' is a list of lists.

    """

    def __init__(self, structure=None, percolating_species=None,
                 cutoff=None, static_species=None, flip_sequence=None,
                 formula_units=1, sublattices=None, bonds=None,
                 sort_sites=False, **kwargs):
        """
        Args:
          The constructor takes as arguments the same keys expected in
          the JSON input file (doc object doc string).

        """

        self.input_dict = kwargs

        if structure is None:
            raise KeyError("No structure file specified.")
        else:
            self.structure_path = structure
            self.input_structure = Poscar.from_file(
                self.structure_path).structure

        if sort_sites:
            idx = np.lexsort(
                np.array([s.coords for s in self.input_structure]).T)
            sites = [self.input_structure[i] for i in idx]
            self.input_structure = Structure.from_sites(sites)

        self.percolating_species = percolating_species

        self.cutoff = cutoff
        self.static_species = [] if static_species is None else static_species
        self.flip_sequence = [] if flip_sequence is None else flip_sequence
        self.formula_units = formula_units

        self.sublattices = {}
        if sublattices is not None:
            for sl in sublattices:
                sl_dict = sublattices[sl]
                self.sublattices[sl] = Sublattice(
                    self.input_structure, **sl_dict)

        self.bonds = {}
        if bonds is not None:
            for b in bonds:
                bond = Bond.from_dict(b)
                # dictionary, so that the following syntax is valid:
                # if Bond('A', 'B') in self.bonds:
                #    rules = self.bonds[Bond('A', 'B')].bond_rules
                self.bonds[bond] = bond

        # reduce structure to sites only from sublattices that are not
        # ignored
        active_sites = []
        for sl in self.sublattices:
            if not self.sublattices[sl].ignore:
                active_sites.extend(self.sublattices[sl].sites)
        active_sites.sort()
        self.structure = Structure.from_sites(
            [self.input_structure[i] for i in active_sites])
        if len(active_sites) < self.input_structure.num_sites:
            inactive_sites = [i for i in range(self.input_structure.num_sites)
                              if i not in active_sites]
            self.static_sites = Structure.from_sites(
                [self.input_structure[i] for i in inactive_sites])
        else:
            self.static_sites = None

        # assign a sublattice label to each site in the reduced structure
        self.site_labels = ["" for i in active_sites]
        for sl in self.sublattices:
            for s in self.sublattices[sl].sites:
                if s in active_sites:
                    self.site_labels[active_sites.index(s)] = sl

    @classmethod
    def from_file(cls, json_file, **kwargs):
        """
        Args:
          json_file   path or file object pointing to JSON input file

        """

        if isinstance(json_file, str):
            with open(json_file) as fp:
                input_dict = json.load(fp)
        else:
            input_dict = json.load(json_file)

        input_dict.update(kwargs)
        return cls(**input_dict)

    @classmethod
    def from_string(cls, json_string, **kwargs):
        """
        Args:
          json_string   string containing JSON input

        """

        input_dict = json.loads(json_string)
        input_dict.update(kwargs)
        return cls(**input_dict)

    def __str__(self):
        out = " Dribble Input\n"
        out += " -------------\n"
        out += "\n Percolating species: " + ", ".join(
            self.percolating_species)
        out += "\n Static species     : " + ", ".join(self.static_species)
        out += "\n Cutoff             : {}".format(self.cutoff)
        out += "\n Formula units      : {}".format(self.formula_units)
        out += "\n Flip sequence      : " + ", ".join(
            ["{} --> {}".format(a, b) for a, b in self.flip_sequence])
        out += "\n Sublattices        : " + ", ".join(
            [s for s in self.sublattices])
        out += "\n Bonds              : " + ", ".join(
            [str(self.bonds[b]) for b in self.bonds])
        return out

    @property
    def mg_structure(self):
        warn("'mg_structure' has been depricated in favor of the "
             "'structure' attribute.",  DeprecationWarning)
        return self.structure

    @property
    def initial_occupancy(self):
        """
        Convenience property returning the initial occupancies of all
        sublattices.

        """
        occup_dict = {sl: self.sublattices[sl].initial_occupancy
                      for sl in self.sublattices
                      if self.sublattices[sl].initial_occupancy is not None}
        return None if occup_dict == {} else occup_dict
