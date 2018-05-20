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
Module with auxiliary functions.

"""

from __future__ import print_function, division, unicode_literals

import sys
import numpy as np


__author__ = "Alexander Urban"
__date__ = "2013-01-22"


def uprint(string, **kwargs):
    """
    This function is basically the regular (Python 3) print function,
    but it does not buffer the output.

    Note: the keyword argument `file' of the print function is
          not available (connected to stdout).
    """
    print(string, **kwargs)
    sys.stdout.flush()


class ProgressBar(object):

    def __init__(self, N, width=80, char=u"\u25ae"):
        """
        Start a simple ASCII progress bar to indicate the progress
        of the MC calculation.  Use together with _print_progress_bar.

        Arguments:
          N (int): Number of samples until 100%
          width (int): Width in characters
          char (char): The character used to draw the progress bar
        """

        uprint(" 0%                25%                 "
               "50%                 75%                 100%")
        uprint(" ", end="")

        self._steps = N
        self._width = width
        self._increment = float(width)/float(N)
        self._char = char[0]
        self._length = 0
        self._buffer = 0.0
        self._count = -1

    def __call__(self):
        """
        Print next mark or finish the progress bar.
        """

        self._count += 1
        self._buffer += self._increment

        if self._count == self._steps:
            uprint(" done.\n")
            return

        if self._count > self._steps:
            return

        if (self._buffer > 1.0):
            N = min(int(self._buffer), self._width-self._length)
            self._buffer -= N
            self._length += N
            uprint(N*self._char, end="")


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
