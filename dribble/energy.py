"""
Tools to investigate the energetic (rather than percolation)
properties of lattices.

"""

from __future__ import print_function, division

import numpy as np
import random
from math import exp

__author__ = "Alexander Urban"
__email__ = "aurban@atomistic.net"
__date__ = "2018-05-19"
__version__ = "0.1"


class Metropolis(object):
    """
    A Metropolis Monte Carlo implementation.  Currently, only
    nearest-neighbor models are available.

    Attributes:
      lattice: Instance of lattice.Lattice
      bonds (dict): Dict with sublattice.Bond for each pair of
          sublattices A and B
      energy (float): The total energy for the present lattice decoration

    Methods:
      bond_energy: Return energy for a specific bond
      site_energy: Returns the energy of a specific sublattice & species
      eval_total_energy: Evaluate the total energy.  Normally not needed
          as self.energy is incrementally updated

    """

    def __init__(self, lattice, sublattices, bonds):
        """
        Args:
          lattice: Instance of lattice.Lattice
          sublattices (dict): Dict of sublattices (sublattice.Sublattice)
          bonds (dict): Dict with sublattice.Bond for each pair of
              sublattices A and B

        """

        self.lattice = lattice
        self.sublattices = sublattices
        self.bonds = bonds
        self.energy = self.eval_total_energy()

    def __str__(self):
        out = " Metropolis Monte-Carlo Simulation\n"
        out += " ---------------------------------\n\n"
        out += " Bonds:\n"
        for b in self.bonds:
            for inter in b.interactions:
                out += "   {} {} V = {}\n".format(
                    b, set(inter), b.interactions[inter])
        out += " Energy: {}\n".format(self.energy)
        return out

    def bond_energy(self, sublattice1, sublattice2, species1, species2):
        """
        Return the bond energy for a bond between two specific sublattices
        and species.

        Args:
          sublattice1, sublattice2 (str): The labels of the two sublattices
          species1, species2 (str): The two species

        Returns:
          energy (float)

        """

        bond = frozenset([sublattice1, sublattice2])
        return self.bonds[bond].energy(species1, species2)

    def site_energy(self, sublattice, species):
        """
        Return the site energy for a specific sublattice and species.

        Args:
          sublattice (str): The label of the sublattice
          species (str): The species

        Returns:
          energy (float)

        """

        return 0.0  # Currently, we do not use site energies

    def eval_total_energy(self):
        """
        Brute-force evaluation of the total energy.  This method should only
        be called during initialization or for comparison with the
        incrementally updated energy.

        Note: The incremental energy stored in self.energy is not
              overwritten by this method, instead the energy is returned.

        Returns:
          energy (float)

        """

        energy = 0.0
        for site1 in range(self.lattice.num_sites):
            sublattice1 = self.lattice.site_labels[site1]
            species1 = self.lattice.species[site1]
            energy += self.site_energy(sublattice1, species1)
            for site2 in self.lattice.nn[site1]:
                sublattice2 = self.lattice.site_labels[site2]
                species2 = self.lattice.species[site2]
                energy += 0.5*self.bond_energy(sublattice1, sublattice2,
                                               species1, species2)
        return energy

    def eval_site_energy(self, site, species=None):
        """
        Evaluate the energy of a specific site.

        Args:
          site (int): The site ID
          species (str): Optionally, the species of the site can be
              specified.  If no species is given, the current species
              will be used

        Returns:
          energy (float)

        """

        sublattice = self.lattice.site_labels[site]
        if species is None:
            species = self.lattice.species[site]
        energy = self.site_energy(sublattice, species)
        for site2 in self.lattice.nn[site]:
            sublattice2 = self.lattice.site_labels[site2]
            species2 = self.lattice.species[site2]
            energy += self.bond_energy(sublattice, sublattice2,
                                       species, species2)
        return energy

    def flip_energy(self, site, species):
        """
        Determine the energy required or gained when a specific site is
        updated to a specific species.

        Note: This method does not actually change the site occupancy.

        Args:
          site (int): The site ID
          species (str): The species

        Returns:
          energy_after_flip - energy_before_flip (float)

        """

        if species == self.lattice.species[site]:
            # nothing has changed
            return 0.0
        else:
            energy_before_flip = self.eval_site_energy(site)
            energy_after_flip = self.eval_site_energy(site, species=species)
            return energy_after_flip - energy_before_flip

    def doubleflip_energy(self, site1, species1, site2, species2):
        """
        Determine the energy required or gained when a two site are
        updated to different species.

        Note: This method does not actually change the site occupancy.

        Args:
          site1, site2 (int): The site IDs
          species1, species2 (str): The species

        Returns:
          energy_after_flip - energy_before_flip (float)

        """

        energy_before_flip = self.eval_site_energy(site1)
        energy_before_flip += self.eval_site_energy(site2)

        energy_after_flip = self.eval_site_energy(site1, species=species1)
        # temporarily change the species of site1, but then revert again
        species1_orig = self.lattice.species[site1]
        self.lattice.species[site1] = species1
        energy_after_flip += self.eval_site_energy(site2, species=species2)
        self.lattice.species[site1] = species1_orig

        return energy_after_flip - energy_before_flip

    def canonical_draw_sites(self):
        """
        Randomly select two sites from the same sublattice but with
        different species.

        """

        site1 = random.randrange(self.lattice.num_sites)
        species1 = self.lattice.species[site1]
        sublattice = self.lattice.site_labels[site1]
        site2 = random.choice([i for i in self.lattice.sublattice(sublattice)
                               if self.lattice.species[i] != species1])
        species2 = self.lattice.species[site2]
        return (site1, species1, site2, species2)

    def grand_canonical_draw_site(self):
        site = random.randrange(self.lattice.num_sites)
        sl = self.lattice.site_labels[site]
        species_orig = self.lattice.species[site]
        species_other = [s for s in self.sublattices[sl].allowed_species
                         if s != species_orig]
        species_new = np.random.choice(species_other)
        return (site, species_orig, species_new)

    def canonical_mc_step(self, beta, num_flips=None):
        """
        Perform a single canonical (N, V, T) Monte Carlo step consisting of
        a series of double flips that does not change the overall
        concentration of any species.

        This method updates the lattice occupancy and the `energy` attribute.

        Args:
          beta (float): Thermodynamic beta, i.e., 1/(kB*T), where kB is
              Boltzmann's constant and T is the temperature.
          num_flips (int): Number of double flips.  If not specified, the
              number of flips will be set to the total number of sites.

        Returns:
          num_accepted/num_flips (float): Fraction of accepted flips

        """

        N = self.lattice.num_sites if num_flips is None else num_flips
        p = np.random.random(N)  # at most N random numbers will be needed
        num_accepted = 0
        for i in range(N):
            (site1, species1, site2, species2) = self.canonical_draw_sites()
            dE = self.doubleflip_energy(site1, species2, site2, species1)
            # accept flip with Metropolis condition
            if (dE < 0.0) or (p[i] <= exp(-dE*beta)):
                self.lattice.set_site(site1, species2)
                self.lattice.set_site(site2, species1)
                self.energy += dE
                num_accepted += 1
        return num_accepted/N

    def grand_canonical_mc_step(self, beta, mu, num_flips=None):
        """
        Perform a single grand-canonical (mu, V, T) Monte Carlo step
        consisting of a series of flips that may change the overall
        concentration of any species.

        This method updates the lattice occupancy, the `energy`
        attribute, and the 'grand_potential' attribute.

        Args:
          beta (float): Thermodynamic beta, i.e., 1/(kB*T), where kB is
              Boltzmann's constant and T is the temperature.
          mu (dict): Chemical potentials for all atomic species.
          num_flips (int): Number of double flips.  If not specified, the
              number of flips will be set to the total number of sites.

        Returns:
          num_accepted/num_flips (float): Fraction of accepted flips

        """

        N = self.lattice.num_sites if num_flips is None else num_flips
        p = np.random.random(N)  # at most N random numbers will be needed
        num_accepted = 0
        for i in range(N):
            (site, species_orig, species_new
             ) = self.grand_canonical_draw_site()
            dE = self.flip_energy(site, species_new)
            dO = dE + mu[species_new] - mu[species_orig]
            # accept flip with Metropolis condition
            if (dO < 0.0) or (p[i] <= exp(-dO*beta)):
                self.lattice.set_site(site, species_new)
                self.energy += dE
                num_accepted += 1
        return num_accepted/N
