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
A class to represent lattices.

"""

from __future__ import print_function, division, unicode_literals

import numpy as np
import sys

from .nblist import NeighborList

__author__ = "Alexander Urban"
__date__ = "2013-02-15"


class Lattice(object):

    def __init__(self, lattice_vectors, frac_coords,
                 supercell=(1, 1, 1), NN_range=None, site_labels=None,
                 species=None, occupying_species=[], static_species=[],
                 concentrations=None, formula_units=1, neighbor_list=None):
        """
        Arguments:
          lattice_vectors    3x3 matrix with lattice vectors in rows
          frac_coords        Nx3 array; fractional coordinates of the
                             N lattice sites
          supercell          list of multiples of the cell in the three
                             spacial directions
          NN_range           cutoff to be used for nearest-neighbor detection
          site_labels        list of labels for each site
          species            list of species for each site
          occupying_species  list of species associated with occupied sites
          static_species     list of species associated with static sites
          concentrations     dict with species concentrations for
                             selected sublattices as dict; requires site labels
                             Example:
                             {"A": {"Li": 0.5, "Vac": 0.5},
                              "B": {"Co": 1.0}}
          formula_units      Number of formula units in input structure
                             (only used to compute compositions)
          neighbor_list      Can re-use the neighbor list from another
                             Lattice object if passed here.

        """

        """                    static data

        _avec[i][j]   j-th component of the i-th lattice vector
        _coo[i][j]    j-th component of the coordinates of the i-th
                      lattice site
        _nsites       total number of lattice sites

        _dNN[i]             nearest neighbor distance from the i-th site
        _nn[i][j]           j-th nearest neighbor site of site i
        _nnn[i][j]          j-th next nearest neighbor site of site i
                            (only computed, if requested)
        _bonds[i][j]        True, if there is a bond between the j-th site
                            and its j-th neighbor
        _T_nn[i][j]         the translation vector belonging to _nn[i][j]
        _T_nnn[i][j]        the translation vector belonging to _nnn[i][j]
                            (only computed, if requested)
        _N_nn               number of nearest neighbors; integer, if all sites
                            are equivalent, else list of intergers
        _N_nnn              number of next nearest neighbors; integer, if all
                            sites are equivalent, else list of intergers
                            (only computed, if requested)
        _nsurface           number of sites at cell boundary
                            (only computed, if requested)
        """

        self._avec = (np.array(lattice_vectors).T * supercell).T
        self._coo = []
        self._occup = []
        self._occupied = []
        self._vacant = []
        self.supercell = supercell
        self.site_labels = []
        self.species = []
        self._static = []

        self.occupying_species = occupying_species

        self.num_formula_units = formula_units
        self.num_formula_units *= supercell[0]*supercell[1]*supercell[2]

        isite = 0
        for i in range(len(frac_coords)):
            coo = np.array(frac_coords[i])
            if site_labels is not None:
                label_i = site_labels[i]
            else:
                label_i = None
            if species is not None:
                species_i = species[i]
            else:
                species_i = None
            if species_i in occupying_species:
                Oi = 1
            else:
                Oi = -1
            static_i = (species_i in static_species)
            for ix in range(supercell[0]):
                for iy in range(supercell[1]):
                    for iz in range(supercell[2]):
                        self._coo.append((coo + [ix, iy, iz]) /
                                         np.array(supercell, dtype=np.float64))
                        self._occup.append(Oi)
                        self.site_labels.append(label_i)
                        self.species.append(species_i)
                        if (Oi > 0):
                            self._occupied.append(isite)
                        else:
                            self._vacant.append(isite)
                        if static_i:
                            self._static.append(isite)
                        isite += 1

        self._coo = np.array(self._coo)
        self._nsites = len(self._coo)
        self._occup = np.array(self._occup)

        if site_labels is not None and concentrations is not None:
            self.random_species_decoration(
                concentrations, occupying_species=occupying_species,
                static=static_species)

        # stats regarding static sites
        self._num_static = len(self._static)
        self._num_sites_not_static = self._nsites - self._num_static
        self._not_static = list(set(range(self.num_sites)) - set(self._static))
        self._static_vacant = list(set(self._static) & set(self._vacant))
        self._static_occupied = list(set(self._static) & set(self._occupied))

        # initialization of the neighbor list
        if neighbor_list:
            self._nblist = neighbor_list._nblist
            self._dNN = neighbor_list._dNN
            self._nn = neighbor_list._nn
            self._nnn = neighbor_list._nnn
            self._T_nn = neighbor_list._T_nn
            self._T_nnn = neighbor_list._T_nnn
            self._N_nn = neighbor_list._N_nn
            self._N_nnn = neighbor_list._N_nnn
            self._nsurface = neighbor_list._nsurface
            self._nbshells = neighbor_list._nbshells
            self._nbshell_dist = neighbor_list._nbshell_dist
        else:
            self._nblist = []
            self._dNN = []
            self._nn = []
            self._nnn = []
            self._T_nn = []
            self._T_nnn = []
            self._N_nn = 0
            self._N_nnn = 0
            self._nsurface = 0
            self._nbshells = []
            self._nbshell_dist = []
            self._build_neighbor_list(r_NN=NN_range)

    @classmethod
    def from_structure(cls, structure, site_labels=None, **kwargs):
        """
        Create a Lattice instance based on the lattice vectors
        defined in a `Structure' object (pymatgen.core.structure).

        Arguments:

          structure       an instance of pymatgen.core.structure.Structure
          site_labels     labels for sublattices; will be determined from
                          structure if not specified

          All keyword arguments of the main constructor are supported.

        """

        avec = structure.lattice.matrix
        coo = structure.frac_coords
        species = np.array([s.specie.symbol for s in structure])
        if site_labels is None:
            site_labels = species.copy()

        lattice = cls(avec, coo, species=species,
                      site_labels=site_labels, **kwargs)

        return lattice

    @classmethod
    def from_input_object(cls, inp, **kwargs):
        """
        Create a Lattice instance based on data in an Input Parameters object.

        Arguments:

          inp (percol.input.Input): Input Parameters object

        """

        lattice = cls.from_structure(
            inp.structure,
            NN_range=inp.cutoff,
            site_labels=inp.site_labels,
            occupying_species=inp.percolating_species,
            static_species=inp.static_species,
            concentrations=inp.initial_occupancy,
            formula_units=inp.formula_units,
            **kwargs)
        return lattice

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        ostr = "\n Lattice and Sites\n"
        ostr += " -----------------\n\n"
        ostr += " Lattice vectors:\n\n"
        for v in self._avec:
            ostr += "   {:12.8f}  {:12.8f}  {:12.8f}\n".format(*v)
        ostr += "\n total number of sites : {}".format(self._nsites)
        ostr += "\n occupied sites        : {} ({} static)".format(
            self.num_occupied, self.num_static_occupied)
        ostr += "\n vacant sites          : {} ({} static)".format(
            self.num_vacant, self.num_static_vacant)
        if type(self._N_nn) == int:
            ostr += "\n number of NNs         : {}".format(self._N_nn)
        else:
            ostr += "\n average number of NNs : {}".format(
                np.sum(self._N_nn)/float(len(self._N_nn)))
        ostr += "\n"
        ostr += str(self._nblist)
        return ostr

    @property
    def nn(self):
        return list(self._nn)

    @property
    def nnn(self):
        return list(self._nnn)

    @property
    def num_occupied(self):
        return len(self._occupied)

    @property
    def num_vacant(self):
        return len(self._vacant)

    @property
    def num_sites(self):
        return self._nsites

    @property
    def occupied(self):
        return list(self._occupied)

    @property
    def vacant(self):
        return list(self._vacant)

    @property
    def num_static(self):
        return self._num_static

    @property
    def num_sites_not_static(self):
        return self._num_sites_not_static

    @property
    def num_static_occupied(self):
        return len(self._static_occupied)

    @property
    def num_static_vacant(self):
        return len(self._static_vacant)

    @property
    def num_vacant_not_static(self):
        return len(self.vacant_not_static)

    @property
    def num_occupied_not_static(self):
        return len(self.occupied_not_static)

    @property
    def static(self):
        return list(self._static)

    @property
    def not_static(self):
        return list(self._not_static)

    @property
    def static_occupied(self):
        return list(self._static_occupied)

    @property
    def static_vacant(self):
        return list(self._static_vacant)

    @property
    def vacant_not_static(self):
        return list(set(self._vacant) - set(self._static))

    @property
    def occupied_not_static(self):
        return list(set(self._occupied) - set(self._static))

    def sublattice(self, sl):
        """
        List of sites for a specific sublattice.

        Args:
          sl    "site label" of the sublattice

        """
        return [i for i in range(self.num_sites) if self.site_labels[i] == sl]

    @property
    def species_list(self):
        """
        Return list of all species on the lattice.

        """
        return list(set(self.species))

    def sites_of_species(self, species):
        """
        List of sites for specific species.

        Args:
          species   a site species string

        """
        return [i for i in range(self.num_sites)
                if self.species[i] == species]

    def concentration(self, species):
        """
        Return concentration of a given species.

        """

        num_species = len(self.sites_of_species(species))
        return float(num_species)/float(self.num_sites)

    @property
    def composition(self):
        """
        Return concentrations of all species on the lattice.

        """
        f = float(self.num_sites)/float(self.num_formula_units)
        comp = {}
        for s in self.species_list:
            comp[s] = self.concentration(s)*f
        return comp

    def set_site(self, site, species):
        """
        Occupy a given site with a given species and update all data
        structures.

        """

        # keep track of "occupied" and "vacant" sites
        if species in self.occupying_species:
            if site in self._vacant:
                del self._vacant[self._vacant.index(site)]
                self._occupied.append(site)
        else:
            if site in self._occupied:
                del self._occupied[self._occupied.index(site)]
                self._vacant.append(site)

        self.species[site] = species

    def check_if_neighbors(self, sites):
        """
        Check whether all sites in a set are neighbors.

        """
        are_neighbors = True
        num_sites = len(sites)
        for i in range(num_sites):
            s1 = sites[i]
            nb1 = self._nn[s1]
            for j in range(i+1, num_sites):
                s2 = sites[j]
                if s2 not in nb1:
                    are_neighbors = False
                    break
        return are_neighbors

    def random_decoration(self, p=0.5, N=None):
        """
        Randomly occupy lattice sites.

        Arguments:
          p    occupation probability
          N    exact number of sites to be occupied

        Note: if specified, N takes precedence over p.
        """

        if not N:
            N = int(np.floor(p*float(self._nsites)))
        N = max(0, min(N, self._nsites))

        idx = np.random.permutation(self._nsites)
        self._occupied = []
        self._vacant = list(range(self._nsites))
        for i in range(N):
            self._occupied.append(idx[i])
            del self._vacant[self._vacant.index(idx[i])]
        self._occup[:] = -1
        self._occup[idx[0:N]] = 1

    def random_species_decoration(self, concentrations,
                                  occupying_species=[], static=[]):
        """
        Randomly occupy lattice sites with species according to
        sublattice-specific concentrations.

        Arguments:
          concentrations   dict with species concentrations for
                           selected sublattices as dict
                           Example:
                           {"A": {"Li": 0.5, "Vac": 0.5},
                            "B": {"Co": 1.0}}
          occupying_species  list of species whose sites are considered
                             occupied
          static           list of species associated with static sites

        Note: if the concentrations cannot be realized exactly, they
              will be approximated by rounding

        """

        self._occupied = []
        self._vacant = list(range(self._nsites))
        self._occup[:] = -1
        self._static = []
        self.species[:] = [None for i in range(self._nsites)]

        for sl in concentrations:
            sites = self.sublattice(sl)
            num_sites = len(sites)
            decoration = []
            for species in concentrations[sl]:
                c = concentrations[sl][species]
                num_sites_species = int(round(c*num_sites))
                c2 = num_sites_species/num_sites
                if (abs(c - c2) > 1.0e-6):
                    sys.stderr.write(
                        "Warning: Could not exactly realize "
                        "concentration of species '{}.'\n".format(species)
                        + "         Requested: {}  Actual: {}\n".format(c, c2)
                        + "         Try larger supercell.\n")
                decoration += [species for i in range(num_sites_species)]
            while len(decoration) < num_sites:
                decoration.append(decoration[-1])
            while len(decoration) > num_sites:
                decoration = decoration[:-1]
            decoration = np.random.permutation(decoration)
            for i, s in enumerate(sites):
                self.species[s] = decoration[i]
                if decoration[i] in occupying_species:
                    self._occupied.append(s)
                    del self._vacant[self._vacant.index(s)]
                    self._occup[s] = 1
                if decoration[i] in static:
                    self._static.append(s)

    def get_nnn_shells(self, dr=0.1):
        """
        Calculate shells of next nearest neighbors and store them in `nnn'.

        Note: This routine requires cells that are actually large enough
              to contain the next-nearest neighbor shells.

        """

        nnn = []
        T_nnn = []
        N_nnn = np.empty(self._nsites, dtype=int)

        try:
            pbcdist = self._nblist.get_pbc_distances_and_translations
            for i in range(self._nsites):
                nn_i = self._nn[i]
                nnnb = set([])
                for j in nn_i:
                    nn_j = self._nn[j]
                    nnnb |= set(nn_j) - set(nn_i) - {i}
                nnnb = list(nnnb)
                (dist, Tvecs) = pbcdist(i, nnnb[0])
                dmin = dist[0]
                nnn_i = []
                T_nnn_i = []
                for j in nnnb:
                    (dist, Tvec) = pbcdist(i, j)
                    for k in range(len(dist)):
                        if (dist[k] < dmin - dr):
                            nnn_i = [j]
                            T_nnn_i = [Tvec[k]]
                            dmin = dist[k]
                        elif (dist[k] <= dmin + dr):
                            nnn_i.append(j)
                            T_nnn_i.append(Tvec[k])
                nnn.append(nnn_i)
                T_nnn.append(T_nnn_i)
                N_nnn[i] = len(nnn_i)
        except IndexError:
            sys.stderr.write("Error: Could not determine next-nearest "
                             "neighbor shells.  Try larger supercell.\n")
            sys.exit()

        self._nnn = nnn
        self._T_nnn = T_nnn
        if (np.all(N_nnn == N_nnn[0])):
            self._N_nnn = int(N_nnn[0])
        else:
            self._N_nnn = N_nnn

    def save_structure(self, file_name="CONTCAR", vacant="V", occupied="O"):
        """
        Save current occupation to an output file in VASP's POSCAR format.
        Relies on `pymatgen' for the file I/O.

        Arguments:
          file_name    name of the output file
          vacant       atomic species to be placed at vacant sites
          occupied     atomic species to be placed at occupied sites

          Note: `vacant` and `occupied` are only used if the species are
                not explicitly defined when the Lattice instance is created.

          Note: Sites are grouped (sorted) by species, so the order of
                sites in the saved POSCAR file is likely different from
                the original input file.

        """

        try:
            from pymatgen.core import Structure
            from pymatgen.io.vasp.inputs import Poscar
        except ImportError:
            sys.stderr.write("Writing POSCAR files requires `pymatgen`.\n")

        if len(self.species) == 0:
            species = [vacant for i in range(self._nsites)]
            for i in self._occupied:
                species[i] = occupied
        else:
            species = self.species[:]

        species = np.array(species)
        idx = np.argsort(species)

        struc = Structure(self._avec, species[idx], self._coo[idx])
        poscar = Poscar(struc)
        poscar.write_file(file_name)

    def _build_neighbor_list(self, r_NN=None, dr=0.1):
        """
        Determine the list of neighboring sites for
        each site of the lattice.  Allow the next neighbor
        distance to vary about `dr'.
        """

        dNN = np.empty(self._nsites)
        N_nn = np.empty(self._nsites, dtype=int)
        nbs = list(range(self._nsites))
        nbshells = list(range(self._nsites))
        nbshelldist = list(range(self._nsites))
        Tvecs = list(range(self._nsites))

        def neighbor_shells(dist, nbl):
            """ Sort neighbors by shells. """
            idx = np.argsort(dist)
            nb_shells = []
            shell_dist = []
            d = 0.0
            shell = []
            for i in idx:
                if dist[i] - dr > d:
                    nb_shells.append(shell)
                    shell_dist.append(d)
                    shell = []
                    d = dist[i]
                shell.append(nbl[i])
            nb_shells.append(shell)
            shell_dist.append(d)
            return (nb_shells[1:], shell_dist[1:])

        nblist = NeighborList(self._coo, lattice_vectors=self._avec,
                              interaction_range=r_NN, tolerance=dr)

        for i in range(self._nsites):
            if r_NN:
                (nbl, dist, T) = nblist.get_neighbors_and_distances(i, dr=dr)
            else:
                (nbl, dist, T) = nblist.get_nearest_neighbors(i, dr=dr)
            Tvecs[i] = T
            dNN[i] = np.min(dist)
            N_nn[i] = len(nbl)
            (nbshells[i], nbshelldist[i]) = neighbor_shells(dist, nbl)
            nbs[i] = nbl

        self._nblist = nblist
        self._nn = nbs
        self._nbshells = nbshells
        self._nbshell_dist = nbshelldist
        self._T_nn = Tvecs
        if (np.all(np.abs(dNN-dNN[0]) < 0.1*dr)):
            self._dNN = np.min(dNN)
        else:
            self._dNN = dNN
        if (np.all(N_nn == N_nn[0])):
            self._N_nn = int(N_nn[0])
        else:
            self._N_nn = N_nn


if (__name__ == "__main__"):

    print("\n FCC 4x4x4 cell (64 sites)")

    avec = np.array([[0.0, 0.5, 0.5],
                     [0.5, 0.0, 0.5],
                     [0.5, 0.5, 0.0]])*5.0

    coo = np.array([[0.0, 0.0, 0.0]])
    lat = Lattice(avec, coo, supercell=(4, 4, 4))

    print(lat)

    print(" checking number of nearest neighbors (12 for FCC) ... ", end="")
    passed = True
    for nn_i in lat.nn:
        N_nn = len(nn_i)
        if (N_nn != 12):
            print(N_nn)
            print(nn_i)
            passed = False
            break
    if passed:
        print("passed.")
    else:
        print("FAILED!")

    print(" checking number of next nearest neighbors "
          "(6 for FCC) ... ", end="")
    lat.get_nnn_shells()
    passed = True
    for nnn_i in lat.nnn:
        N_nnn = len(nnn_i)
        if (N_nnn != 6):
            print(N_nnn)
            print(nnn_i)
            passed = False
            break
    if passed:
        print("passed.")
    else:
        print("FAILED!")

    print(" testing random decoration of 16 (of 64) sites ... ", end="")
    lat.random_decoration(p=0.25)
    N1 = np.sum(np.where(lat._occup > 0, 1, 0))
    lat.random_decoration(N=16)
    N2 = np.sum(np.where(lat._occup > 0, 1, 0))
    if N1 == N2 == 16:
        print("passed.")
    else:
        print("FAILED.")

    print(" exporting structure to file CONTCAR ... ", end="")
    try:
        lat.save_structure('CONTCAR')
        print("passed.")
    except:
        print("FAILED.")

    print("")
