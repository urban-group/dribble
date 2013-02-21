#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-02-18"

import numpy as np

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
        self._v1 = -4.0*J1
        self._v2 = -4.0*J2
        
        # total energy
        self._E_tot = self.total_energy()

    def __str__(self):
        ostr  = " \nAn instance of the Ising class\n\n"
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
        E = -s_i*(self._J1_2*E_nn + self._J2_2*E_nnn + self._H)

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
        Perform a single MC step at temperature T.
        """

        idx_i = np.random.random_integers(0,len(self._occupied)-1)
        idx_j = np.random.random_integers(0,len(self._vacant)-1)

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
                print(1)

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

#----------------------------------------------------------------------#
#                              unit test                               #
#----------------------------------------------------------------------#

if __name__=="__main__": #{{{ unit test 

    from lattice import Lattice
    
    print("\n FCC 4x4x4 cell (64 sites)\n")
#
#    avec = np.array([ [0.0, 0.5, 0.5],
#                      [0.5, 0.0, 0.5],
#                      [0.5, 0.5, 0.0] ])*5.0
#
#    coo = np.array([[0.0, 0.0, 0.0]])

    avec = np.array([ [0.000000,  1.859700,  1.859700],
                      [1.859700, -1.859700,  0.000000],
                      [3.719400,  1.859700, -1.859700] ])

    coo = np.array([ [0.747322, 0.497478, 0.748766], 
                     [0.247693, 0.497298, 0.248750] ])

    lat = Lattice(avec, coo, supercell=(4,4,4))
    lat.get_nnn_shells()
    lat.random_decoration(p=0.5)

    lat.save_structure('CONTCAR.0')

    print(" number of occupied sites: {}\n".format(lat.num_occupied))

    J1 = 1.0
    J2 = 1.0

    T = 800000.0
    kT_inv = 1.0/(T*k_B)
    tau = 1.0
    NMC = 10000

    ising = IsingModel(lat, J1, J2, H=0.0)

    print(" Running MC simulation ({} steps)".format(NMC))
    print(" (output is written to file `mcsteps.dat')")

    with open('mcsteps.dat', 'w') as f:
        for i in xrange(NMC):
            E = ising.mc_NVT(kT_inv=kT_inv, tau=tau)
            f.write(" {:6d}  {}\n".format(i,E))

    print(" final energy = {}".format(E))
    E_tot = ising.total_energy()
    print(" total energy = {} (calculated from scratch)".format(E_tot))
    if (abs(E-E_tot)>0.1):
        print(" Error: energies inconsistent\n")

    print("\n number of occupied sites: {}".format(lat.num_occupied))

    lat.save_structure('CONTCAR.1')

    print("")
