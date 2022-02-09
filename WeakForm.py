'''
=======================================
Defines weak forms.
(adapted from perturbed_netwon repo
 see https://bitbucket.org/johannrudi/perturbed_newton.git)

Author:     Melody Shih
=======================================
'''

import firedrake as fd
import numpy as np
import math
import sys

import Abstract

QUAD_DEG=15

def delta(u):
	tau_u   = tau(u)
	delta  = 2*fd.inner(tau_u,tau_u)
	return delta

def eta(u, A, H, delta_min, Pstar):
	e = 2
	tau_u   = tau(u)
	delta  = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	return 1.0/e**2*P/2/delta

def update_va(mx, my, t, X, v_a, T, L):
	tday = t*T/24/60/60
	if tday <= 4:
		a = 72./180*np.pi
		ws = -math.tanh((4-tday)*(4+tday)/2)
		#ws = -math.tanh(tday*(8.0-tday)/2.0)
		vmax = 15*ws #*T/L #m/s
		mx.assign(256*1000/L+51.2*1000/(24*60*60)*t*T/L)
		my.assign(256*1000/L+51.2*1000/(24*60*60)*t*T/L)
	else:
		a = 81./180*np.pi
		ws = math.tanh((12-tday)*(tday-4)/2)
		#ws = -math.tanh(tday*(8.0-tday)/2.0)
		vmax = 15*ws #*T/L #m/s
		mx.assign(665.6*1000/L-51.2*1000/(24*60*60)*t*T/L)
		my.assign(665.6*1000/L-51.2*1000/(24*60*60)*t*T/L)

	r = fd.sqrt((mx - X[0])**2 + (my - X[1])**2)
	s = 1.0/50*fd.exp(-r/(100*1000)*L)
	v_a.interpolate(fd.as_vector([s*fd.Constant(vmax)*( fd.cos(a)*(X[0]-mx) + fd.sin(a)*(X[1]-my)),
	                              s*fd.Constant(vmax)*(-fd.sin(a)*(X[0]-mx) + fd.cos(a)*(X[1]-my))]))

def tau_atm(C_a, rho_a, v_a):
	return C_a*rho_a*fd.sqrt(fd.inner(v_a, v_a))*v_a

def tau_ocean(C_o, rho_o, u, v_o):
	return C_o*rho_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*(v_o-u)

def tau(u):
	e = 2
	E = fd.sym(fd.nabla_grad(u))
	I = fd.Identity(2)
	return 1./e*fd.dev(E) + 0.5*fd.tr(E)*I

def applyBoundaryConditions(u, boundary_condition, boundary_condition_type=None):
    assert boundary_condition is not None or boundary_condition_type is not None

    # apply given boundary conditions
    if boundary_condition is not None:
        if isinstance(boundary_condition, fd.DirichletBC):
            boundary_condition.apply(u.vector())
        else:
            for bc in boundary_condition:
                bc.apply(u.vector())
        return

    # create and apply boundary conditions
    if boundary_condition_type is not None:
        bc = createBoundaryConditions(boundary_condition_type, u.function_space())
        applyBoundaryConditions(u, bc)

def sigmaI(u, A, H, delta_min, Pstar):
	tau_u = tau(u)
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	E = fd.sym(fd.nabla_grad(u))
	EI = fd.tr(E)
	return 1./2*(EI/delta-1.0)

def sigmaII(u, A, H, delta_min, Pstar):
	e = 2
	tau_u = tau(u)
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	E = fd.sym(fd.nabla_grad(u))
	EII = 2*fd.sqrt(-fd.det(fd.dev(E)))
	return 1./2/delta/e**2*EII

#=======================================
# Objective
#=======================================
def objective(u, uprev, A, H, FncSp, rho_i, dt, C_a, rho_a, v_a, C_o,
              rho_o, v_o, delta_min, Pstar, f_c):
	'''
	Creates the weak form for the objective functional:

	where
	'''
	tau_u   = tau(u)
	delta  = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	E = fd.sym(fd.nabla_grad(u))
	I = fd.Identity(2)
	obj_divsigma = dt*P/2*delta*fd.dx(degree=QUAD_DEG) - dt*0.5*P*fd.tr(E)*fd.dx(degree=QUAD_DEG)
	obj_rhoHu = 0.5*rho_i*H*fd.inner(u, u)*fd.dx(degree=QUAD_DEG)

	tau_a = tau_atm(C_a, rho_a, v_a)
	tau_o = tau_ocean(C_o, rho_o, u, v_o)
	obj_F =   rho_i*H*fd.inner(uprev, u)*fd.dx(degree=QUAD_DEG)\
	             + dt*fd.inner(tau_a, u)*fd.dx(degree=QUAD_DEG)

	obj = obj_rhoHu + obj_divsigma - obj_F

	if (abs(C_o)>1e-15):
		obj_ocean = dt*(-1./3)*C_o*rho_o*fd.sqrt(fd.inner(v_o-u, v_o-u))**3*fd.dx(degree=QUAD_DEG)
		obj -= obj_ocean

	return obj

#=======================================
# Linearization
#=======================================

def gradient(u, uprev, A, H, FncSp, rho_i, dt, C_a, rho_a, v_a, C_o, rho_o, v_o,
             delta_min, Pstar, f_c):
	'''
	Creates the weak form for the gradient (nonlinear residual):

	    F(ute) - A(u, ute) =
	    \int rho_ice*H_n*uprev*ute + dt*tau_atm(t_n)*ute +
	                               + dt*rho_ice*H_n*f_c*(e_r x v_ocean)*ute
	    -
	    \int rho_ice*H_n*u*ute + dt*rho_ice*H_n*f_c*(e_r x u)*ute
	                           + dt*sigma_n(A_n,H_n,u)*grad(ute)
	                           - dt*tau_ocean(t_n, u)*ute
	where
	    ute = u_test = TestFunction of the velocity space
	'''
	ute = fd.TestFunction(FncSp)

	tau_a = tau_atm(C_a, rho_a, v_a)
	tau_o = tau_ocean(C_o, rho_o, u, v_o)

	tau_u   = tau(u)
	tau_ute = tau(ute)
	Ete = fd.sym(fd.nabla_grad(ute))

	P  = Pstar*H*fd.exp(-20*(1.0-A))
	F  = rho_i*H*fd.inner(uprev, ute)*fd.dx(degree=QUAD_DEG) + \
	          dt*fd.inner(tau_a, ute)*fd.dx(degree=QUAD_DEG)

	AA = rho_i*H*fd.inner(u, ute)*fd.dx(degree=QUAD_DEG)\
	     + dt*P/fd.sqrt(delta_min**2 + 2*fd.inner(tau_u, tau_u))*fd.inner(tau_u, tau_ute)*fd.dx(degree=QUAD_DEG)\
	     - dt*0.5*P*fd.tr(Ete)*fd.dx(degree=QUAD_DEG)

	if (abs(C_o) > 1e-15):
		AA += -dt*fd.inner(tau_o, ute)*fd.dx(degree=QUAD_DEG)

	# Coriolis:
	if (abs(f_c) > 1e-15):
	    er_x_vo = fd.as_vector([-v_o[1], v_o[0]])
	    er_x_u  = fd.as_vector([  -u[1],   u[0]])
	    F  += dt*rho_i*H*f_c*fd.inner(er_x_vo, ute)*fd.dx(degree=QUAD_DEG)
	    AA += dt*rho_i*H*f_c*fd.inner(er_x_u , ute)*fd.dx(degree=QUAD_DEG)


	grad = AA - F

	return grad

def nonlinearres_NewtonStressvel(u, uprev, V, S, Vd, A, H, rho_i, dt, C_a, rho_a,
                                 v_a, C_o, rho_o, v_o, delta_min, Pstar, f_c, maxmag, steplen,
                                 u_perturb):
	'''
	Creates the weak form for the nonlinear residual:
	TODO
	'''
	tau_u        = tau(u)
	tau_uperturb = tau(u_perturb)
	delta        = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	delta_sq     = delta_min**2+2*fd.inner(tau_u,tau_u)
	scale = fd.conditional( fd.lt(fd.inner(S, S), maxmag*maxmag), 1.0, maxmag/fd.sqrt(fd.inner(S,S)))
	S_perturb = - S\
	            - 2.0/delta_sq*scale*fd.inner(tau_uperturb, tau_u)*S\
	            + 1.0/delta*(tau_uperturb+tau_u)
	#S_perturb = - 2.0/delta_sq*fd.inner(tau_uperturb, tau_u)*S*scale\
	#            + 1.0/delta*(tau_uperturb)
	
	ute = fd.TestFunction(V)
	
	tau_a = tau_atm(C_a, rho_a, v_a)
	tau_o = tau_ocean(C_o, rho_o, u+steplen*u_perturb, v_o)
	
	tau_ute = tau(ute)
	Ete = fd.sym(fd.nabla_grad(ute))
	
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	F  = rho_i*H*fd.inner(uprev, ute)*fd.dx(degree=QUAD_DEG) + \
	          dt*fd.inner(tau_a, ute)*fd.dx(degree=QUAD_DEG)
	
	AA = rho_i*H*fd.inner(u, ute)*fd.dx(degree=QUAD_DEG)\
	     + dt*P*fd.inner(S, tau_ute)*fd.dx(degree=QUAD_DEG)\
	     - dt*0.5*P*fd.tr(Ete)*fd.dx(degree=QUAD_DEG)
	
	AA +=  steplen*rho_i*H*fd.inner(u_perturb, ute)*fd.dx(degree=QUAD_DEG)\
	     + steplen*   dt*P*fd.inner(S_perturb, tau_ute)*fd.dx(degree=QUAD_DEG)\
	
	if (abs(C_o) > 1e-15):
	  AA += -dt*fd.inner(tau_o, ute)*fd.dx(degree=QUAD_DEG)
	
	return F - AA

def nonlinearres_NewtonStressvel_notau(u, uprev, V, S, Vd, A, H, rho_i, dt, C_a, rho_a,
                                       v_a, C_o, rho_o, v_o, delta_min, Pstar, f_c, maxmag,
                                       steplen, u_per):
	'''
	Creates the weak form for the nonlinear residual without using aux. var. tau:
	TODO
	'''
	e = 2.0
	einv = 0.5
	esqinv = 0.25
	I = fd.Identity(2)

	grad_u = fd.nabla_grad(u)
	E      = fd.sym(grad_u)
	delta        = fd.sqrt(delta_min**2+2.*esqinv*fd.inner(fd.dev(E),fd.dev(E))+fd.tr(E)*fd.tr(E))
	delta_sq     =         delta_min**2+2.*esqinv*fd.inner(fd.dev(E),fd.dev(E))+fd.tr(E)*fd.tr(E)
	scale = fd.conditional( fd.lt(fd.inner(S, S), maxmag*maxmag), 1.0, maxmag/fd.sqrt(fd.inner(S,S)))
	Eper = fd.sym(fd.nabla_grad(u_per))

    # Without symmetrization
	#S_per = - S\
	#        - 2.0/delta_sq*fd.inner(esqinv*fd.dev(Eper)+0.5*fd.tr(Eper)*I, grad_u)*S*scale\
	#        + 1.0/delta*(1./e*fd.dev(Eper) + 0.5*fd.tr(Eper)*I)\
	#        + 1.0/delta*(1./e*fd.dev(E)    + 0.5*fd.tr(E)*I)
	#S_per = - 2.0/delta_sq*fd.inner(esqinv*fd.dev(Eper)+0.5*fd.tr(Eper)*I, grad_u)*S*scale\
	#        + 1.0/delta*(1./e*fd.dev(Eper) + 0.5*fd.tr(Eper)*I)

    # With symmetrization
	S_per = - S\
	        - 1.0/delta_sq*(esqinv*fd.inner(fd.dev(Eper),fd.dev(E))+0.5*fd.tr(Eper)*fd.tr(E))*S*scale\
	        - 1.0/delta_sq*(1./e*fd.inner(S*scale,fd.dev(Eper))+0.5*fd.tr(S*scale)*fd.tr(Eper))*\
	                                                              (1./e*fd.dev(E) + 0.5*fd.tr(E)*I)\
	        + 1.0/delta*(1./e*fd.dev(Eper) + 0.5*fd.tr(Eper)*I)\
	        + 1.0/delta*(1./e*fd.dev(E)    + 0.5*fd.tr(E)*I)
	#S_per = - 1.0/delta_sq*(esqinv*fd.inner(fd.dev(Eper),fd.dev(E))+0.5*fd.tr(Eper)*fd.tr(E))*S*scale\
	#        - 1.0/delta_sq*(1./e*fd.inner(S*scale,fd.dev(Eper))+0.5*fd.tr(S*scale)*fd.tr(Eper))*\
	#                                                              (1./e*fd.dev(E) + 0.5*fd.tr(E)*I)\
	#        + 1.0/delta*(1./e*fd.dev(Eper) + 0.5*fd.tr(Eper)*I)
	
	ute      = fd.TestFunction(V)
	grad_ute = fd.nabla_grad(ute)
	
	tau_a = tau_atm(C_a, rho_a, v_a)
	
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	F  = rho_i*H*fd.inner(uprev, ute)*fd.dx(degree=QUAD_DEG) +\
	          dt*fd.inner(tau_a, ute)*fd.dx(degree=QUAD_DEG)
	
	AA = rho_i*H*fd.inner(u, ute)*fd.dx(degree=QUAD_DEG)\
	     + dt*P*fd.inner(einv*S+0.5*(1-einv)*fd.tr(S)*I, grad_ute)*fd.dx(degree=QUAD_DEG)\
	     - dt*fd.inner(0.5*P*I, grad_ute)*fd.dx(degree=QUAD_DEG)
	
	AA +=  steplen*rho_i*H*fd.inner(u_per, ute)*fd.dx(degree=QUAD_DEG)\
	     + steplen*dt*P*fd.inner(einv*S_per+0.5*(1-einv)*fd.tr(S_per)*I,grad_ute)*fd.dx(degree=QUAD_DEG)
	
	if (abs(C_o) > 1e-15):
		tau_o = tau_ocean(C_o, rho_o, u+steplen*u_per, v_o)
		AA += -dt*fd.inner(tau_o, ute)*fd.dx(degree=QUAD_DEG)
	
	return F - AA

def hessian_NewtonStandard(u, A, H, FncSp, rho_i, dt, C_a, rho_a, v_a, C_o,
	                       rho_o, v_o, delta_min, Pstar, f_c):
	'''
	Creates the weak form for the Hessian of the standard Newton linearization:

	    A'(u)(w, ute) = #TODO: need to update

	where
	    utr = u_trial = TrialFunction of the velocity space
	    ute = u_test = TestFunction of the velocity space
	'''
	utr = fd.TrialFunction(FncSp)
	ute = fd.TestFunction(FncSp)
	er_x_utr  = fd.as_vector([  -utr[1],   utr[0]])

	hess = rho_i*H*fd.inner(utr,ute)*fd.dx(degree=QUAD_DEG)

	# d(sigma)/d(u)
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	tau_u   = tau(u)
	tau_ute = tau(ute)
	tau_utr = tau(utr)
	delta  = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	dsigmadu =         P/delta*fd.inner(tau_utr,tau_ute)*fd.dx(degree=QUAD_DEG) + \
	           -(P/delta**3)*2*fd.inner(tau_u  ,tau_utr)*fd.inner(tau_u,tau_ute)*fd.dx(degree=QUAD_DEG)

	hess += dt*dsigmadu

	# dtau_ocean/du
	if (abs(C_o) > 1e-15):
		dtauodu = -rho_o*C_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(utr,ute)*fd.dx(degree=QUAD_DEG) + \
	    	      -rho_o*C_o/fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(v_o-u,utr)*fd.inner(v_o-u,ute)*fd.dx(degree=QUAD_DEG)
		hess -= dt*dtauodu

	# Coriolis:
	if (abs(f_c) > 1e-15):
	    hess += dt*rho_i*H*f_c*fd.inner(er_x_utr,ute)*fd.dx(degree=QUAD_DEG)

	return hess

def hessian_NewtonStressvel(u, S, A, H, FncSp, rho_i, dt, C_a, rho_a, v_a, C_o,
	                       rho_o, v_o, delta_min, Pstar, f_c, maxmag):
	'''
	Creates the weak form for the Hessian of the stress-vel Newton linearization:

	#TODO: need to update

	'''
	utr = fd.TrialFunction(FncSp)
	ute = fd.TestFunction(FncSp)
	er_x_utr  = fd.as_vector([  -utr[1],   utr[0]])

	hess = rho_i*H*fd.inner(utr,ute)*fd.dx(degree=QUAD_DEG)

	# d(sigma)/d(u)
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	tau_u   = tau(u)
	tau_ute = tau(ute)
	tau_utr = tau(utr)
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	scale = fd.conditional( fd.lt(fd.inner(S, S), maxmag*maxmag), 1.0, maxmag/fd.sqrt(fd.inner(S,S)))
	dsigmadu =         P/delta*fd.inner(tau_utr,tau_ute)*fd.dx(degree=QUAD_DEG)\
	           -(P/delta**2)*2*fd.inner(tau_u  ,tau_utr)*fd.inner(S*scale,tau_ute)*fd.dx(degree=QUAD_DEG)

	hess += dt*dsigmadu

	# dtau_ocean/du
	if (abs(C_o) > 1e-15):
		dtauodu = -rho_o*C_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(utr,ute)*fd.dx(degree=QUAD_DEG) + \
	    	      -rho_o*C_o/fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(v_o-u,utr)*fd.inner(v_o-u,ute)*fd.dx(degree=QUAD_DEG)
		hess -= dt*dtauodu

	# Coriolis:
	if (abs(f_c) > 1e-15):
	    hess += rho_i*H*f_c*fd.inner(er_x_utr,ute)*fd.dx(degree=QUAD_DEG)

	return hess

def hessian_NewtonStressvel_Sym(u, S, A, H, FncSp, rho_i, dt, C_a, rho_a, v_a, C_o,
                         rho_o, v_o, delta_min, Pstar, f_c, maxmag):
	'''
	Creates the weak form for the Hessian of the stress-vel Newton linearization:
	
	#TODO: need to update
	
	'''
	utr = fd.TrialFunction(FncSp)
	ute = fd.TestFunction(FncSp)
	er_x_utr  = fd.as_vector([  -utr[1],   utr[0]])
	
	hess = rho_i*H*fd.inner(utr,ute)*fd.dx(degree=QUAD_DEG)
	
	# d(sigma)/d(u)
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	tau_u   = tau(u)
	tau_ute = tau(ute)
	tau_utr = tau(utr)
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	scale = fd.conditional( fd.lt(fd.inner(S, S), maxmag*maxmag), 1.0, maxmag/fd.sqrt(fd.inner(S,S)))
	dsigmadu =         P/delta*fd.inner(tau_utr,tau_ute)*fd.dx(degree=QUAD_DEG)\
	           -(P/delta**2)*2*0.5*(fd.inner(tau_u,tau_utr)*fd.inner(S*scale,tau_ute)+fd.inner(S*scale,tau_utr)*fd.inner(tau_u,tau_ute))*fd.dx(degree=QUAD_DEG)
	
	hess += dt*dsigmadu
	
	# dtau_ocean/du
	if (abs(C_o) > 1e-15):
	  dtauodu = -rho_o*C_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(utr,ute)*fd.dx(degree=QUAD_DEG) + \
	            -rho_o*C_o/fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(v_o-u,utr)*fd.inner(v_o-u,ute)*fd.dx(degree=QUAD_DEG)
	  hess -= dt*dtauodu
	
	# Coriolis:
	if (abs(f_c) > 1e-15):
	    hess += rho_i*H*f_c*fd.inner(er_x_utr,ute)*fd.dx(degree=QUAD_DEG)
	
	return hess


def hessian_dualStep(u, ustep, S, DualFncSp, delta_min):
	'''
	Creates the weak form for step of dual variable
	'''
	Ste = fd.TestFunction(DualFncSp)

	tau_u     = tau(u)
	tau_ustep = tau(ustep)
	delta     = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	delta_sq  = delta_min**2+2*fd.inner(tau_u,tau_u)
	S_step    = - fd.inner(S,Ste)*fd.dx(degree=QUAD_DEG)\
              - 2.0/delta_sq*fd.inner(tau_ustep, tau_u)*fd.inner(S,Ste)*fd.dx(degree=QUAD_DEG)\
              + 1.0/delta*fd.inner(tau_ustep+tau_u, Ste)*fd.dx(degree=QUAD_DEG)
	return S_step

def hessian_dualStep_Sym(u, ustep, S, DualFncSp, delta_min):
	'''
	Creates the weak form for step of dual variable
	'''
	Ste = fd.TestFunction(DualFncSp)

	tau_u     = tau(u)
	tau_ustep = tau(ustep)
	delta     = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	delta_sq  = delta_min**2+2*fd.inner(tau_u,tau_u)
	S_step    = - fd.inner(S,Ste)*fd.dx(degree=QUAD_DEG)\
              - 2.0/delta_sq*fd.inner(tau_ustep, tau_u)*fd.inner(S,Ste)*fd.dx(degree=QUAD_DEG)\
              + 1.0/delta*fd.inner(tau_ustep+tau_u, Ste)*fd.dx(degree=QUAD_DEG)
	return S_step

def dualresidual(S, u, DualFncSp, delta_min):
	'''
	Creates the weak form for residual of dual variable
	'''
	Ste   = fd.TestFunction(DualFncSp)
	tau_u = tau(u)
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	res   = delta*fd.inner(S, Ste)*fd.dx(degree=QUAD_DEG) - fd.inner(tau_u, Ste)*fd.dx(degree=QUAD_DEG)
	return res

def hessian_dualUpdate_boundMaxMagnitude(S, DualFncSp, max_magn):
	S_test = fd.TestFunction(DualFncSp)
	S_rescaled = fd.conditional( fd.lt(fd.inner(S, S), max_magn*max_magn),\
	                fd.inner(S, S_test),\
	                fd.inner(S, S_test)/fd.sqrt(fd.inner(S,S))*max_magn)*fd.dx(degree=QUAD_DEG)
	S_ind = fd.conditional( fd.lt(fd.inner(S, S), max_magn*max_magn),\
	                0.0,\
	                1.0)*fd.dx(degree=QUAD_DEG)
	return S_rescaled, S_ind
