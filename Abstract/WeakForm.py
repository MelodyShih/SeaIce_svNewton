'''
=======================================
Defines abstract weak forms.

Author:             Johann Rudi
=======================================
'''

import firedrake as fd
import sys

#=======================================
# Basic Weak Forms
#=======================================

def mass(FncSp):
    '''
    Creates the weak form for a mass matrix.
    '''
    return fd.inner(fd.TrialFunction(FncSp), fd.TestFunction(FncSp)) * fd.dx

def magnitude(U, FncSpScalar):
    '''
    Creates the weak form for computing a pointwise magnitude.
    '''
    return fd.sqrt(fd.inner(U, U)) * fd.TestFunction(FncSpScalar) * fd.dx

def magnitude_normalize(U, FncSp):
    '''
    Creates the weak form for normalizing a function by its magnitude.
    '''
    magn     = fd.sqrt(fd.inner(U, U))
    magn_min = 1.0e20 * sys.float_info.min
    return fd.inner( fd.conditional(fd.lt(magn_min, magn), U/magn, 0.0*U),
                     fd.TestFunction(FncSp) ) * fd.dx

def magnitude_scale(U_magn, U_scal, FncSp):
    '''
    Creates the weak form for scaling a function by the magnitude of another function.
    '''
    magn = fd.sqrt(fd.inner(U_magn, U_magn))
    return fd.inner( fd.conditional(fd.lt(1.0, magn), magn*U_scal, U_scal),
                     fd.TestFunction(FncSp) ) * fd.dx
