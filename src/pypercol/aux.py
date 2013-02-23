#!/usr/bin/env python

"""
Insert description here.
"""

from __future__ import print_function

__author__ = "Alexander Urban"
__date__   = "2013-01-22"

import sys
import os

import numpy as np

#----------------------------------------------------------------------#
#                           unbuffered print                           #
#----------------------------------------------------------------------#

unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
def uprint(string, **kwargs):
    """
    This function is basically the regular (Python 3) print function, 
    but it does not buffer the output.

    Note: the keyword argument `file' of the print function is
          not available (connected to stdout).
    """
    print(string, file=unbuffered, **kwargs)

#----------------------------------------------------------------------#
#                          ASCII progress bar                          #
#----------------------------------------------------------------------#

class ProgressBar(object):

    def __init__(self, N, char=u"\u25ae"):
        """
        Start a simple ASCII progress bar to indicate the progress
        of the MC calculation.  Use together with _print_progress_bar.

        Arguments:
          N     number of samples until 100%
          char  the character used to draw the progress bar
        """
        
        uprint( " 0%                25%                 "
              + "50%                 75%                 100%" )
        uprint(" ", end="")

        self._steps  = N
        self._nprint = int(max(round(float(N)/80.0), 1))
        self._nchar  = int(max(round(80.0/float(N)), 1))
        self._char   = char[0]
        self._count  = -1

    def __call__(self):
        """
        Print next mark or finish the progress bar.
        """

        self._count += 1

        if self._count == self._steps:
            uprint(" done.\n")
            return

        if self._count > self._steps:
            return

        if (self._count % self._nprint == 0):
            uprint(self._nchar*self._char.encode("utf-8"), end="")

    


#----------------------------------------------------------------------#
#                              functions                               #
#----------------------------------------------------------------------#

def sigmoidal1(x, x0, y0, A, c):
    """
    Returns a sigmoidal function based on the hyperbolic tangent.

    A*tanh(b*(x-x0)) + y0

    Arguments:
      x0     x-coordinate of the turning point
      y0     y-coordinate of the turning point
      A      Amplitude
      c      contraction factor
    """

    return A*np.tanh(c*(x-x0)) + y0

def sigmoidal2(x, x0, y0, A, c):
    """
    Returns a modified sigmoid function.

    A/(1.0 + np.exp(-c*(x-x0))) + y0

    Arguments:
      x0     x-coordinate of the turning point
      y0     y-coordinate of the turning point
      A      Amplitude
      c      contraction factor
    """
    
    return A/(1.0 + np.exp(-c*(x-x0))) + y0


