#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-02-18"

import numpy as np
import sys

# k_B = 1.3806505e-23  # J/K
k_B = 3.1668157e-06  # Ha/K

class IsingModel(object):

    def __init__(self, lattice, J1, J2, H=1.0):

        # keep references to lattice
        self._lattice  = lattice
        self._nsites   = lattice._nsites
        self._nn       = lattice._nn
        self._nnn      = lattice._nnn
        self._occup    = lattice._occup
        self._occupied = lattice._occupied
        self._vacant   = lattice._vacant

        # we work with J/2 since that's what we need all the time
        self._J1_2 = 0.5*J1
        self._J2_2 = 0.5*J2
        self._H    = H

        # translation to lattice gas Hamiltonian speech
        self._v1 =  4.0*J1
        self._v2 =  4.0*J2
        
        # total energy
        self._E_tot = self.total_energy()

        self.niter = 0
        self.nacc  = 0
        self.nrej  = 0

    def __str__(self):
        ostr  = "\n Instance of the Ising class:\n\n"
        ostr += " nearest neighbor interaction:      J1 = {}\n".format(
            self._J1_2*2.0)
        ostr += " next nearest neighbor interaction: J2 = {}\n".format(
            self._J2_2*2.0)
        ostr += " point term (magnetic field):       H  = {}\n".format(
            self._H)
        ostr += "\n"
        return ostr

    def __repr__(self):
        return self.__str__()
    
    #------------------------------------------------------------------#
    #                          public methods                          #
    #------------------------------------------------------------------#

    def E(self, i):
        """
        Energy of site i.
        """

        E_nn = 0
        for j in self._nn[i]:
            E_nn += np.sign(self._occup[j])
        E_nnn = 0
        for j in self._nnn[i]:
            E_nnn += np.sign(self._occup[j])
        s_i = np.sign(self._occup[i])
        E = s_i*(self._J1_2*E_nn + self._J2_2*E_nnn + self._H)

        return E 

    def total_energy(self):
        """
        Total energy of the structure.
        """

        E_tot = 0.0
        for i in xrange(self._nsites):
            E_tot += self.E(i)

        return E_tot
        
    def mc_NVT(self, kT_inv=1000.0, tau=1.0):
        """
        Perform a single MC step at temperature T with kT_inv = 1/(k*T).
        Canonical ensemble, i.e., fixed particle number.
        """

        self.niter += 1

        noccupied = len(self._occupied)
        nvacant   = self._nsites - noccupied

        for istep in xrange(self._nsites):

            idx_i = np.random.random_integers(0,noccupied-1)
            idx_j = np.random.random_integers(0,nvacant-1)
            
            i = self._occupied[idx_i]
            j = self._vacant[idx_j]
            
            # affected sited
            sites  = [i,j] + self._nn[i] + self._nn[j] 
            sites += self._nnn[i] + self._nnn[j]
            sites = set(sites)
            
            dE = 0.0
            for k in sites:
                dE -= self.E(k)
            self._occup[i] = -1
            self._occup[j] =  1
            for k in sites:
                dE += self.E(k)
            
            accept = False
            if (dE <= 0):
                accept = True
            else:
                r = np.random.random()
                if (r <= tau*np.exp(-kT_inv*dE)):
                    accept = True
                    self.nacc += 1
                else:
                    self.nrej += 1
            
            if accept:
                self._E_tot += dE
                del self._occupied[idx_i]
                del self._vacant[idx_j]
                self._occupied.append(j)
                self._vacant.append(i)
            else:
                self._occup[i] =  1
                self._occup[j] = -1

        return self._E_tot

    def mc_mVT(self, kT_inv=1000.0, tau=1.0):
        """
        Perform a single MC step at temperature T with kT_inv = 1/(k*T).
        Grand canonical ensemble, i.e, fixed chemical potential, 
        but variable particle number.
        """

        self.niter += 1

        for istep in xrange(self._nsites):

            i = np.random.random_integers(0,self._nsites-1)
            
            if self._occup[i] > 0:
                idx_i = self._occupied.index(i)
            else:
                idx_i = self._vacant.index(i)
            
            # affected sited
            sites  = [i] + self._nn[i] + self._nnn[i] 
            sites = set(sites)
            
            dE = 0.0
            for k in sites:
                dE -= self.E(k)
            self._occup[i] = -np.sign(self._occup[i])
            for k in sites:
                dE += self.E(k)
            
            accept = False
            if (dE <= 0):
                accept = True
            else:
                r = np.random.random()
                if (r <= tau*np.exp(-kT_inv*dE)):
                    accept = True
                    self.nacc += 1
                else:
                    self.nrej += 1
            
            if accept:
                self._E_tot += dE
                if self._occup[i] < 0: # then it was > 0 before flip
                    idx_i = self._occupied.index(i)
                    del self._occupied[idx_i]
                    self._vacant.append(i)
                else:
                    idx_i = self._vacant.index(i)
                    del self._vacant[idx_i]
                    self._occupied.append(i)
            else:
                # flip back
                self._occup[i] = -np.sign(self._occup[i])

        return self._E_tot

#----------------------------------------------------------------------#
#                              unit test                               #
#----------------------------------------------------------------------#

if __name__=="__main__": #{{{ unit test 

    from lattice import Lattice
    
    print("\n FCC, 50% occupied,  J1 = J2\n")

    # starting from layered structure
    avec = np.array([ [0.000000,  1.859700,  1.859700],
                      [1.859700, -1.859700,  0.000000],
                      [3.719400,  1.859700, -1.859700] ])

    coo = np.array([ [0.747322, 0.497478, 0.748766], 
                     [0.247693, 0.497298, 0.248750] ])

    lat = Lattice(avec, coo, decoration=[1, -1], supercell=(4,4,4))
    lat.get_nnn_shells()

    lat.save_structure('CONTCAR.0')

    print(" number of occupied sites: {}\n".format(lat.num_occupied))

    J1 = 0.5e-3
    J2 = 0.5e-3

    T = 350.0
    kT_inv = 1.0/(T*k_B)
    tau = 1.0
    NMC = 10000

    #------------------------------------------------------------------#
    #                        layered structure                         #
    #------------------------------------------------------------------#

    ising = IsingModel(lat, J1, J2, H=0.0)
    E0 = ising.total_energy()
    print(" energy of the layered structure = {}".format(E0))

    #------------------------------------------------------------------#
    #                      FCC, J1 == J2, H == 0                       #
    #------------------------------------------------------------------#

    J1 = 1.0
    J2 = 1.0

    T = 700000.0
    kT_inv = 1.0/(T*k_B)
    tau = 1.0
    NMC = 10000

    # re-initialize with random decoration:
    lat.random_decoration(p=0.5)
    ising = IsingModel(lat, J1, J2, H=0.0)

    print("\n Test 1: repulsive NN and NNN interactions, NVT ensemble\n")

    print(" Running MC simulation ({} steps)".format(NMC))
    print(" (output is written to file `mcsteps1.dat')")

    with open('mcsteps1.dat', 'w') as f:
        for i in xrange(NMC):
            E = ising.mc_NVT(kT_inv=kT_inv, tau=tau)
            f.write(" {:6d}  {}\n".format(i,E))

    print(" final energy = {}".format(E))
    E_tot = ising.total_energy()
    print(" total energy = {} (calculated from scratch)".format(E_tot))
    if (abs(E-E_tot)>0.1):
        print("\n Error: energies inconsistent!")

    print(" accepted/rejected: {}".format(float(ising.nacc)/float(ising.nrej)))

    print("\n number of occupied sites: {}".format(lat.num_occupied))

    lat.save_structure('CONTCAR.1')

    sys.exit()

    #------------------------------------------------------------------#
    #                       NN only (attractive)                       #
    #------------------------------------------------------------------#

    print("\n Test 2: only attractive NN interactions (no NNN) mVT ensemble\n")

    J1 = -1.0
    J2 =  0.0
    H  = 1.0

    T = 700000.0
    kT_inv = 1.0/(T*k_B)
    tau = 1.0
    NMC = 2000

    # re-initialize with random decoration:
    lat.random_decoration(p=0.5)
    ising = IsingModel(lat, J1, J2, H=H)

    print(" Running MC simulation ({} steps)".format(NMC))
    print(" (output is written to file `mcsteps2.dat')")

    with open('mcsteps2.dat', 'w') as f:
        for i in xrange(NMC):
            E = ising.mc_mVT(kT_inv=kT_inv, tau=tau)
            f.write(" {:6d}  {}\n".format(i,E))

    print(" final energy = {}".format(E))
    E_tot = ising.total_energy()
    print(" total energy = {} (calculated from scratch)".format(E_tot))
    if (abs(E-E_tot)>0.1):
        print("\n Error: energies inconsistent!")
    print(" ground state = {}".format(128*6*J1 - abs(128*H)))

    print(" accepted/rejected: {}".format(float(ising.nacc)/float(ising.nrej)))

    print("\n number of occupied sites: {}".format(lat.num_occupied))

    lat.save_structure('CONTCAR.2')

    #------------------------------------------------------------------#
    #                       NN only (repulsive)                        #
    #------------------------------------------------------------------#

    print("\n Test 3: only repulsive NN interactions (no NNN) mVT ensemble\n")

    J1 = 5.0
    J2 = 0.0
    H  = 3.0

    T = 450000.0
    kT_inv = 1.0/(T*k_B)
    tau = 1.0
    NMC = 10000

    # re-initialize with random decoration:
    lat.random_decoration(p=0.5)
    ising = IsingModel(lat, J1, J2, H=H)

    print(" Running MC simulation ({} steps)".format(NMC))
    print(" (output is written to file `mcsteps3.dat')")

    with open('mcsteps3.dat', 'w') as f:
        for i in xrange(NMC):
            E = ising.mc_mVT(kT_inv=kT_inv, tau=tau)
            f.write(" {:6d}  {}\n".format(i,E))

    print(" final energy = {}".format(E))
    E_tot = ising.total_energy()
    print(" total energy = {} (calculated from scratch)".format(E_tot))
    if (abs(E-E_tot)>0.1):
        print("\n Error: energies inconsistent!")
#    print(" ground state = {}".format(128*6*J1 - abs(128*H)))

    print(" accepted/rejected: {}".format(float(ising.nacc)/float(ising.nrej)))

    print("\n number of occupied sites: {}".format(lat.num_occupied))

    lat.save_structure('CONTCAR.3')

    print("")
