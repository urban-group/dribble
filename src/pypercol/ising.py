#!/usr/bin/env python

"""
Insert description here.
"""

__author__ = "Alexander Urban"
__date__   = "2013-02-18"

import numpy as np

class IsingModel(object):

    def __init__(self, lattice):

        self._lattice = lattice
        self._nn      = lattice._nn
        self._T_nn    = lattice._T_nn
        self._nnn     = lattice._nnn
        self._T_nnn   = lattice._T_nnn

    def __str__(self):
        ostr  = " \nAn instance of the Ising class\n\n"
        return ostr

    def __repr__(self):
        return self.__str__()
    
