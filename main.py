#!/usr/bin/python -t
# -*- coding: utf-8 -*-
import datetime
import numpy as np
from scipy import linalg

"""
FEM - Finite Element Method 
"""

__author__="Tetsuo Koyama<tkoyama010@gmail.com>"
__status__="production"
__version__="0.0.1"
__date__=datetime.date.today().isoformat()

def P(r, s):
  return np.array([[ -0.5**2*(1.-s),  0.5**2*(1.-s), 0.5**2*(1.+s), -0.5**2*(1.+s) ],\
                   [ -0.5**2*(1.-r), -0.5**2*(1.+r), 0.5**2*(1.+r),  0.5**2*(1.-r) ]])

def X(x1, x2, x3, x4, y1, y2, y3, y4):
  return np.array([[ x1, y1 ],\
                   [ x2, y2 ],\
                   [ x3, y3 ],\
                   [ x4, y4 ]])

def J(r, s, iso2D):
  return np.dot(P( r = r, s = s ), X( x1 = iso2D.x1, y1 = iso2D.y1,\
                                      x2 = iso2D.x2, y2 = iso2D.y2,\
                                      x3 = iso2D.x3, y3 = iso2D.y3,\
                                      x4 = iso2D.x4, y4 = iso2D.y4))

def B(gauss, iso2D):
  B = np.dot(np.linalg.inv(J(r = gauss.r, s = gauss.s, iso2D = iso2D)), P(r = gauss.r, s = gauss.s))
  return np.array([[ B[0, 0], 0.     , B[0, 1], 0.     , B[0, 2], 0.     , B[0, 3], 0.     ],\
                   [ 0.     , B[1, 0], 0.     , B[1, 1], 0.     , B[1, 2], 0.     , B[1, 3]],\
                   [ B[1, 0], B[0, 0], B[1, 1], B[0, 1], B[1, 2], B[0, 2], B[1, 3], B[0, 3]]])

def D(young, poisson):
  C1 = 1.  - poisson
  C2 = poisson
  C3 = 0.5 - poisson
  return (young/(1. + poisson)/(1. - 2.*poisson))*np.array([[ C1, C2, 0.],\
                                              [ C2, C1, 0.],\
                                              [ 0., 0., C3]])

def Ke(gauss, iso2D):
  return iso2D.T*np.dot(np.dot(B(gauss= gauss, iso2D = iso2D).T,\
                               D(young = iso2D.young, poisson = iso2D.poisson)),  \
                               B(gauss= gauss, iso2D = iso2D))  \
        *np.linalg.det(J(r = gauss.r , s = gauss.s, iso2D = iso2D))

class iso2D:

  '''
  Element
  '''

  def __init__(self, T, young, poisson, x1, x2, x3, x4, y1, y2, y3, y4):

    self.T  = T

    self.young  = young
    self.poisson = poisson

    self.x1 = x1
    self.x2 = x2
    self.x3 = x3
    self.x4 = x4
    self.y1 = y1
    self.y2 = y2
    self.y3 = y3
    self.y4 = y4

class gauss:
  '''
  Gauss's integration
  '''

  def __init__(self, r, s):

    self.r = r
    self.s = s

class node:
  '''
  node
  '''

  def __init__(self, x, y):

    self.x = x
    self.y = y

if __name__ == "__main__":
  '''
  input data
  '''
  np.set_printoptions(linewidth = 'Inf', precision = 4)
  print 'Ex 3.3'
  iso2D3 = iso2D(T = 1., young = 1., poisson = 0., x1 = 0., x2 = 2., x3 = 2., x4 = 0., y1 = 0., y2 = 0., y3 = 1., y4 = 2.)
  gauss3 = gauss(r = 0.0, s = 0.0)
  print Ke(gauss= gauss3, iso2D = iso2D3) * 48.
  print 'Ex 3.4'
  iso2D4 = iso2D(T = 1., young = 1., poisson = 0., x1 = 0., x2 = 1., x3 = 1., x4 = 0., y1 = 0., y2 = 1., y3 = 2., y4 = 2.)
  gauss4 = gauss(r = 0.0, s = 0.0)
  print Ke(gauss= gauss4, iso2D = iso2D4) * 48.

