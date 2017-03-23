"""
Parse JSON input files with percolation rules, site labels, etc.

"""

from __future__ import print_function, division
import json
import numpy as np

__author__ = "Alexander Urban"
__email__ = "aurban@lbl.gov"
__date__ = "2017-02-14"
__version__ = "0.1"


class Input(object):

    def __init__(self, input_dict):
        """
        Args:
           input_dict  dictionary with input instructions

        """

        self.input_dict = input_dict

        if "structure" in self.input_dict:
            self.structure_path = self.input_dict["structure"]
        else:
            raise KeyError("No structure file specified.")
        if "percolating_species" in self.input_dict:
            self.percolating_species = self.input_dict["percolating_species"]
        else:
            raise KeyError("No percolating species specified.")

        self.cutoff = None
        self.sublattices = {}
        self.bonds = {}
        self.static_species = []
        self.initial_occupancy = {}
        self.flip_sequence = []
        self.formula_units = 1

        if "cutoff" in self.input_dict:
            self.cutoff = self.input_dict["cutoff"]
        if "sublattices" in self.input_dict:
            self.sublattices = self.input_dict["sublattices"]
        if "bonds" in self.input_dict:
            # bonds are "sets" as the order of sites does not matter
            self.bonds = [set(b) for b in self.input_dict["bonds"]]
        if "static_species" in self.input_dict:
            self.static_species = self.input_dict["static_species"]
        if "initial_occupancy" in self.input_dict:
            self.initial_occupancy = self.input_dict["initial_occupancy"]
        if "flip_sequence" in self.input_dict:
            self.flip_sequence = self.input_dict["flip_sequence"]
        if "formula_units" in self.input_dict:
            self.formula_units = self.input_dict["formula_units"]

    @classmethod
    def from_file(cls, json_file):
        """
        Args:
          json_file   path or file object pointing to JSON input file

        """

        if isinstance(json_file, str):
            with open(json_file) as fp:
                input_dict = json.load(fp)
        else:
            input_dict = json.load(json_file)

        return cls(input_dict)

    @classmethod
    def from_string(cls, json_string):
        """
        Args:
          json_string   string containing JSON input

        """

        input_dict = json.loads(json_string)
        return cls(input_dict)

    def __str__(self):
        return

    @property
    def mg_structure(self):
        from pymatgen.io.vasp import Poscar
        return Poscar.from_file(self.structure_path).structure

    @property
    def site_labels(self):
        """
        List of sublattice labels for each site in the order of the input
        structure.

        """
        sites = []
        labels = []
        for s in self.sublattices:
            s_sites = self.sublattices[s]["sites"]
            sites += s_sites[:]
            labels += [s for i in s_sites]
        idx = np.argsort(sites)
        return [labels[i] for i in idx]
