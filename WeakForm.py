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

def update_va(mx, my, t, X, v_a, T, L):
	a = 72./180*np.pi
	vmax = 15*T/L #m/s
	mx.assign(256*1000/L+128*1000*t*T/L)
	my.assign(256*1000/L+128*1000*t*T/L)
	r = fd.sqrt((mx - X[0])**2 + (my - X[1])**2) 
	s = 1/50*fd.exp(-r/(100*1000)*L)
	v_a.interpolate(fd.as_vector([-s*vmax*( fd.cos(a)*(X[0]-mx) + fd.sin(a)*(X[1]-my)), 
	                              -s*vmax*(-fd.sin(a)*(X[0]-mx) + fd.cos(a)*(X[1]-my))]))

def tau_atm(C_a, rho_a, v_a):
	return C_a*rho_a*fd.sqrt(fd.inner(v_a, v_a))*v_a

def tau_ocean(C_o, rho_o, u, v_o):
	return C_o*rho_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*(v_o-u)

def tau(u):
	e = 2
	E = fd.sym(fd.nabla_grad(u))
	I = fd.Identity(2)
	return 1./e*fd.dev(E) + 0.5*fd.tr(E)*I

#=======================================
# Objective 
#=======================================
def objective(u, uprev, A, H, FncSp, rho_i, delta_t, C_a, rho_a, v_a, C_o, 
              rho_o, v_o, delta_min, Pstar, f_c):
	'''
	Creates the weak form for the objective functional:
	
	where
	'''
	tau_u   = tau(u)
	delta  = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	obj_divsigma = delta_t* P/2*delta*fd.dx 
	obj_rhoHu = 0.5*rho_i*H*fd.inner(u, u)*fd.dx

	tau_a = tau_atm(C_a, rho_a, v_a)
	tau_o = tau_ocean(C_o, rho_o, u, v_o)
	#er_x_vo = fd.as_vector([-v_o[1], v_o[0]])
	obj_F =   rho_i*H*fd.inner(uprev, u)*fd.dx\
	        + delta_t*fd.inner(tau_a, u)*fd.dx#\
	        #+ delta_t*f_c*fd.inner(er_x_vo, u)*fd.dx 

	obj = obj_rhoHu + obj_divsigma - obj_F

	if (abs(C_o)>1e-15):
		obj_ocean = 1./3*C_o*rho_o*fd.sqrt(fd.inner(v_o-u, v_o-u))**3*fd.dx
		obj += obj_ocean

	return obj
        
#=======================================
# Linearization
#=======================================

def gradient(u, uprev, A, H, FncSp, rho_i, delta_t, C_a, rho_a, v_a, C_o, rho_o, v_o, 
             delta_min, Pstar, f_c):
	'''
	Creates the weak form for the gradient:
	
	    F(ute) - A(u, ute) = 
	    \int rho_ice*H_n*uprev*ute + delta_t*tau_atm(t_n)*ute + 
	                               + delta_t*rho_ice*H_n*h_c*(e_r x v_ocean)*ute
	    - 
	    \int rho_ice*H_n*u*ute + delta_t*rho_ice*H_n*f_c*(e_r x u)*ute
	                           + delta_t*sigma_n(A_n,H_n,u)*grad(ute)
	                           - delta_t*tau_ocean(t_n, u)*ute
	where
	    ute = u_test = TestFunction of the velocity space
	'''
	ute = fd.TestFunction(FncSp)
	    
	tau_a = tau_atm(C_a, rho_a, v_a)
	tau_o = tau_ocean(C_o, rho_o, u, v_o)

	er_x_vo = fd.as_vector([-v_o[1], v_o[0]])
	er_x_u  = fd.as_vector([  -u[1],   u[0]])

	tau_u   = tau(u)
	tau_ute = tau(ute)
	Ete = fd.sym(fd.nabla_grad(ute))

	P  = Pstar*H*fd.exp(-20*(1.0-A))
	F  = rho_i*H*fd.inner(uprev, ute)*fd.dx + \
	     delta_t*fd.inner(tau_a, ute)*fd.dx#\
	     #+ delta_t*rho_i*H*f_c*fd.inner(er_x_vo, ute)*fd.dx 

	AA = rho_i*H*fd.inner(u, ute)*fd.dx\
	     + delta_t*P/fd.sqrt(delta_min**2 + 2*fd.inner(tau_u, tau_u))*fd.inner(tau_u, tau_ute)*fd.dx\
	     - delta_t*P*fd.tr(Ete)*fd.dx#\
	     #+ delta_t*rho_i*H*f_c*fd.inner(er_x_u , ute)*fd.dx

	if (abs(C_o) > 1e-15):
		AA += - delta_t*fd.inner(tau_o, ute)*fd.dx	

	grad = AA - F

	return grad

def hessian_NewtonStandard(u, A, H, FncSp, rho_i, delta_t, C_a, rho_a, v_a, C_o, 
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

	hess =         rho_i*H*fd.inner(utr,ute)*fd.dx#\
	       #delta_t*rho_i*H*f_c*fd.inner(er_x_utr,ute)*fd.dx

	# d(sigma)/d(u)
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	tau_u   = tau(u)
	tau_ute = tau(ute)
	tau_utr = tau(utr) 
	delta  = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	dsigmadu =   P/delta*fd.inner(tau_utr,tau_ute)*fd.dx + \
               -(P/delta**3)*2*fd.inner(tau_u,tau_utr)*fd.inner(tau_u,tau_ute)*fd.dx 

	hess += delta_t*dsigmadu

	# dtau_ocean/du
	if (abs(C_o) > 1e-15):
		dtauodu = -rho_o*C_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(utr,ute)*fd.dx + \
	    	      -rho_o*C_o/fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(v_o-u,utr)*fd.inner(v_o-u,ute)*fd.dx
		hess += delta_t*dtauodu

           
	return hess

def hessian_NewtonStressvel(u, S, A, H, FncSp, rho_i, delta_t, C_a, rho_a, v_a, C_o, 
	                       rho_o, v_o, delta_min, Pstar, f_c):
	'''
	Creates the weak form for the Hessian of the stress-vel Newton linearization:
	
	#TODO: need to update

	'''
	utr = fd.TrialFunction(FncSp)
	ute = fd.TestFunction(FncSp)
	er_x_utr  = fd.as_vector([  -utr[1],   utr[0]])

	hess =         rho_i*H*fd.inner(utr,ute)*fd.dx #+ \
	       #delta_t*rho_i*H*f_c*fd.inner(er_x_utr,ute)*fd.dx

	# d(sigma)/d(u)
	P  = Pstar*H*fd.exp(-20*(1.0-A))
	tau_u   = tau(u)
	tau_ute = tau(ute)
	tau_utr = tau(utr) 
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	dsigmadu =         P/delta*fd.inner(tau_utr,tau_ute)*fd.dx\
               -(P/delta**2)*2*fd.inner(tau_u,tau_utr)*fd.inner(S,tau_ute)*fd.dx 
               #-(P/delta**2)*2*(fd.inner(tau_u,tau_utr)*fd.inner(S,tau_ute) + fd.inner(S,tau_utr)*fd.inner(tau_u,tau_ute))*fd.dx 

	hess += delta_t*dsigmadu

	# dtau_ocean/du
	if (abs(C_o) > 1e-15):
		dtauodu = -rho_o*C_o*fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(utr,ute)*fd.dx + \
	    	      -rho_o*C_o/fd.sqrt(fd.inner(v_o-u, v_o-u))*fd.inner(v_o-u,utr)*fd.inner(v_o-u,ute)*fd.dx
		hess += delta_t*dtauodu
           
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
	S_step    = - fd.inner(S,Ste)*fd.dx\
                - 2.0/delta_sq*fd.inner(tau_ustep, tau_u)*fd.inner(S,Ste)*fd.dx\
                + 1.0/delta*fd.inner(tau_ustep+tau_u, Ste)*fd.dx
	return S_step

def dualresidual(S, u, DualFncSp, delta_min):
	'''
	Creates the weak form for residual of dual variable 
	'''
	Ste   = fd.TestFunction(DualFncSp)
	tau_u = tau(u)
	delta = fd.sqrt(delta_min**2+2*fd.inner(tau_u,tau_u))
	res   = delta*fd.inner(S, Ste)*fd.dx - fd.inner(tau_u, Ste)*fd.dx
	return res

def hessian_dualUpdate_boundMaxMagnitude(S, DualFncSp, max_magn):
	S_test = fd.TestFunction(DualFncSp)
	S_rescaled = fd.conditional( fd.lt(fd.inner(S, S), max_magn*max_magn),\
	                fd.inner(S, S_test),\
	                fd.inner(S, S_test)/fd.sqrt(fd.inner(S,S))*max_magn)*fd.dx
	return S_rescaled
