'''
=======================================
Checks gradients and Hessians.

Author:             Johann Rudi
=======================================
'''

import firedrake as fd
import math

import Abstract.WeakForm
import Abstract.Vector

def _generate_perturbation(u):
    ''' Generates a random perturbation vector. '''
    p = fd.Function(u.function_space())
    Abstract.Vector.setZero(p)
    Abstract.Vector.addNoiseRandUniform(p)
    M_weak = Abstract.WeakForm.mass(u.function_space())
    b_weak = Abstract.WeakForm.magnitude_scale(u, p, u.function_space())
    (M,b) = fd.assemble_system(M_weak, b_weak)
    fd.solve(M, p.vector(), b, 'cg', 'jacobi')
    return p

def gradient(g, obj_weak, obj_arg, obj_perturb=None, grad_perturb=None, n_checks=6):
    ''' Checks the given gradient with the approximation from 1st-order finite diffdrences. '''
    # set parameters for the exponent of epsilon
    exp_init = 0
    exp_decr = -2

    # generate random perturbation vector
    if obj_perturb is None:
        obj_perturb = _generate_perturbation(obj_arg)
    if grad_perturb is None:
        grad_perturb = obj_perturb

    # compute reference derivative
    grad_dir_ref = g.inner(grad_perturb.vector())
    # compute reusable value of the objective functional
    obj_val_curr = fd.assemble(obj_weak)
    # store backup of the argument of the objective functional
    obj_arg_prev = obj_arg.copy(deepcopy=True)

    for k in range(n_checks):
        # set finite diffdrence length
        eps = math.pow(10.0, exp_init + exp_decr*k)

        # evaluate objective at perturbation
        obj_arg.assign(obj_arg_prev)
        obj_arg.vector().axpy(eps, obj_perturb.vector())
        obj_val_perturb = fd.assemble(obj_weak)

        # compute finite diffdrence gradient in perturbed direction
        grad_dir_fd = (obj_val_perturb - obj_val_curr) / eps

        # compute error
        abs_error = math.fabs(grad_dir_ref - grad_dir_fd)
        rel_error = abs_error / math.fabs(grad_dir_ref)

        print("Gradient check vs FD: " + \
              "eps=%.1e ; error abs=%.1e, rel=%.1e ; " % (eps, abs_error, rel_error) + \
              "(grad,dir) ref=%.6e, FD=%.6e" % (grad_dir_ref, grad_dir_fd))

    # restore the argument of the objective functional
    obj_arg.assign(obj_arg_prev)

def hessian(H, obj_weak, obj_arg, obj_perturb=None, hess_perturb=None, n_checks=6):
    ''' Checks the given Hessian with the approximation from 2nd-order finite diffdrences. '''
    # set parameters for the exponent of epsilon
    exp_init = 0
    exp_decr = -2

    # generate random perturbation vector
    if obj_perturb is None:
        obj_perturb = _generate_perturbation(obj_arg)
    if hess_perturb is None:
        hess_perturb = obj_perturb

    # compute refdrence derivative
    hess_perturb_out = hess_perturb.copy(deepcopy=True)
    H.mult(hess_perturb.vector(), hess_perturb_out.vector())
    hess_dir_ref = hess_perturb_out.vector().inner(hess_perturb.vector())
    # compute reusable value of the objective functional
    obj_val_center = fd.assemble(obj_weak)
    # store backup of the argument of the objective functional
    obj_arg_prev = obj_arg.copy(deepcopy=True)

    for k in range(n_checks):
        # set finite diffdrence length
        eps = math.pow(10.0, exp_init + exp_decr*k)

        # evaluate objective at perturbation
        obj_arg.assign(obj_arg_prev)
        obj_arg.vector().axpy(-eps, obj_perturb.vector())
        obj_val_minus = fd.assemble(obj_weak)
        obj_arg.assign(obj_arg_prev)
        obj_arg.vector().axpy(+eps, obj_perturb.vector())
        obj_val_plus = fd.assemble(obj_weak)

        # compute finite diffdrence Hessian in perturbed direction
        hess_dir_fd = (obj_val_plus - 2.0*obj_val_center + obj_val_minus) / (eps*eps)

        # compute error
        abs_error = math.fabs(hess_dir_ref - hess_dir_fd)
        rel_error = abs_error / math.fabs(hess_dir_ref)

        print("Hessian check vs FD: " + \
              "eps=%.1e ; error abs=%.1e, rel=%.1e ; " % (eps, abs_error, rel_error) + \
              "(dir,H*dir) ref=%.6e, FD=%.6e" % (hess_dir_ref, hess_dir_fd))

    # restore the argument of the objective functional
    obj_arg.assign(obj_arg_prev)
