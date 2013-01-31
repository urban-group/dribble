#!/usr/bin/env python

"""
Insert description here.
"""

__author__ = "Alexander Urban"
__date__   = "2013-01-22"

import numpy as np
#import scipy.special
from scipy.stats import binom

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

def binom_conv(fn, n, N, p):
    """
    Binomial convolution of a data series.
           __   
           \    / N \    n        N-n
    f(p) = /_  |     |  p  (1 - p)    f(n)
            n   \ n /

    Arguments:
      fn    list of function values f(n_i)
      n     list of discrete nodes {n_i} where 0 <= n_i <= N
      N     maximum value of n_i
      p     continous value to evaluate the distribution at
    """
    
    fp = 0.0
    pdf = binom(N, p)
    for i in range(len(n)):
        fp += pdf.pmf(n)*fn[i]
#        fp += scipy.special.binom(N,n[i])*p**n[i]*(1.0-p)**(N-n[i])*fn[i]
    return fp

