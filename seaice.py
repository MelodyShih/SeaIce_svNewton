from firedrake import *
import math
import WeakForm
import Abstract.Vector
import Abstract.WeakForm
import Abstract.CheckDerivatives

PETSc.Sys.popErrorHandler()
##################################################
##
## Nonlinear eq:
##  A(v_n, phi) = F(phi) - A(v_{n-1}, phi)
##  A(v_n, phi) = (rho_ice H_n v_n, phi_v) + k (rho_ice H_n f_c e_r x v_n, phi_v)
##                                         + k (sigma(v_n, H_n, A_n, grad(phi)
##                                         - k (tau_ocean(t_n, v_n), phi)
##  F(phi) = (rho_ice H_n v_{n-1}, phi_v) + k (tau_atm(t_n), phi) 
##                                        + k (rho_ice H_n f_c e_r x v_ocean, phi_v)
##
##################################################

import argparse
import numpy as np
PETSc.Sys.popErrorHandler()

import logging
#logging.basicConfig(level="INFO")

#======================================
# Parsing input arguments
#======================================

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--linearization", type=str, default="stdnewton")
args, _ = parser.parse_known_args()

#======================================
## nolinear solver parameters
NL_CHECK_GRADIENT = False
MONITOR_NL_ITER = True
MONITOR_NL_STEPSEARCH = False
NL_SOLVER_GRAD_RTOL = 1e-4
NL_SOLVER_GRAD_STEP_RTOL = 1e-8
NL_SOLVER_MAXITER = 300
NL_SOLVER_STEP_MAXITER = 15
NL_SOLVER_STEP_ARMIJO = 1.0e-4
nonlin_it_total = 0

## Output
OUTPUT_VTK = True
OUTPUT_VTK_INIT = False
if OUTPUT_VTK:
    #outfile_va  = File("vtk/"+args.linearization+"/sol_va_ts.pvd")
    outfile_eta = File("vtk/"+args.linearization+"/sol_eta_ts.pvd")
    #outfile_A   = File("vtk/"+args.linearization+"/sol_A_ts.pvd")
    #outfile_H   = File("vtk/"+args.linearization+"/sol_H_ts.pvd")
    #outfile_u   = File("vtk/"+args.linearization+"/sol_u_ts.pvd")

## Scaling
T = 1e3       # 1e3 second
L = 1e3       # 1e3 m
G = 1         # 1 m

## Domain: (0,1)x(0,1)
dim = 2
Nx = Ny = 128
Lx = Ly = 512*1000/L
mesh = RectangleMesh(Nx, Ny, Lx, Ly)
PETSc.Sys.Print("[info] dx in km", 512/Nx)

Tfinal = 3*24*60*60/T #2/T #days
dt  = 30*60/T/40 # 30 min.
dtc = Constant(dt)
t   = 0.0
ntstep = 0
PETSc.Sys.Print("[info] Tfinal in days", Tfinal)
PETSc.Sys.Print("[info] dt in days", dt*T)

## Discretization: 
# v: sea ice velocity (P_k)
# A: sea ice concentration (P_k-2)
# H: mean sea ice thickness (P_k-2)
velt  = FiniteElement("CG", mesh.ufl_cell(), 1)
vdelt = TensorElement("DG", mesh.ufl_cell(), 2)
Aelt  = FiniteElement("DG", mesh.ufl_cell(), 2)
Helt  = FiniteElement("DG", mesh.ufl_cell(), 2)

V = VectorFunctionSpace(mesh, velt)
Vd= FunctionSpace(mesh, vdelt)
Vd1 = FunctionSpace(mesh, "DG", 2)
A = FunctionSpace(mesh, Aelt)
H = FunctionSpace(mesh, Helt)

## Parameteres
# fc: Coriolis
fc = 0.0 #1.46e-4*T #s^{-1}
# air drag coeff.
Ca = 1.2e-3*L/G 
# water drag coeff.
Co = 5.5e-3*L/G
# air density
rhoa = 1.3/900.0 #kg/m^3 
rhoo = 1026.0/900.0 #kg/m^3
rhoice = 900.0/900.0 #kg/m^3
# ice strength 
Pstar = 27.5e3*T**2/L**2/900.0
#other
delta_min = 2e-9*T

# solution vector
sol_u      = Function(V)
sol_uprev  = Function(V)
sol_uprevt = Function(V)
step_u     = Function(V)
sol_A      = Function(A)
step_A     = Function(A)
sol_H      = Function(H)
step_H     = Function(H)
eta        = Function(Vd1)

# other vector
v_ocean = Function(V)
v_a     = Function(V)

## Boundary condition
bc_u     = [DirichletBC(V, Constant((0.,)*dim), "on_boundary")]
bcstep_u = [DirichletBC(V, Constant((0.,)*dim), "on_boundary")]

## Initialization
X = SpatialCoordinate(mesh)
# v_0 = 0.0
sol_u.project(Constant((0.,)*dim))
#sol_u.interpolate(as_vector([(2*X[1]-Lx)/Lx,-(2*X[0]-Ly)/Ly]))
sol_uprevt.assign(sol_u)
# A_0 = 1.0
#bell_r0 = 32; bell_x0 = 256; bell_y0 = 256+96
#bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(X[0]-bell_x0, 2) + pow(X[1]-bell_y0, 2))/bell_r0, 1.0)))
#sol_A.interpolate(1+bell)
sol_A.project(Constant(1.0))


# H_0
#sol_H.interpolate((0.3/L + 0.005/L*(sin(60*X[0]/1e6*L) + sin(30*X[1]/1e6*L)))/G)
sol_H.interpolate((0.3 + 0.005*(sin(60*X[0]/1e6*L) + sin(30*X[1]/1e6*L)))/G)

# v_ocean
v_ocean_max = 0.01*T/L #m/s
v_ocean.interpolate(as_vector([v_ocean_max*(2*X[1]-Lx)/Lx,-v_ocean_max*(2*X[0]-Ly)/Ly]))

# v_a
mx = Constant(0.0)
my = Constant(0.0)
WeakForm.update_va(mx, my, 0.0, X, v_a, T, L)

# visualize initial value
if OUTPUT_VTK_INIT:
    File("vtk/"+args.linearization+"/initial_vo.pvd").write(v_ocean)
    File("vtk/"+args.linearization+"/initial_va.pvd").write(v_a)
    File("vtk/"+args.linearization+"/initial_H.pvd").write(sol_H)

## Weak Form
# set weak forms of objective functional and gradient
obj  = WeakForm.objective(sol_u, sol_uprevt, sol_A, sol_H, V, rhoice, 20*dt, Ca, rhoa, v_a, \
                         Co, rhoo, v_ocean, delta_min, Pstar, fc)
grad = WeakForm.gradient(sol_u, sol_uprevt, sol_A, sol_H, V, rhoice, 20*dt, Ca, rhoa, v_a, \
                         Co, rhoo, v_ocean, delta_min, Pstar, fc)

# set weak form of Hessian and forms related to the linearization
if args.linearization == 'stdnewton':
    hess = WeakForm.hessian_NewtonStandard(sol_u, sol_A, sol_H, V, rhoice, \
               20*dt, Ca, rhoa, v_a, Co, rhoo, v_ocean, delta_min, Pstar, fc)
elif args.linearization == 'stressvel':
    if Vd is None:
        raise ValueError("stressvel not implemented for discretisation %s" \
                                   % vvstokesprob.discretisation)
    S      = Function(Vd)
    S_step = Function(Vd)
    S_proj = Function(Vd)
    S_prev = Function(Vd)

    dualStep = WeakForm.hessian_dualStep(sol_u, step_u, S, Vd, delta_min)
    dualres = WeakForm.dualresidual(S, sol_u, Vd, delta_min)
    hess = WeakForm.hessian_NewtonStressvel(sol_u, S_proj, sol_A, sol_H, V, rhoice, \
               20*dt, Ca, rhoa, v_a, Co, rhoo, v_ocean, delta_min, Pstar, fc)
else:
    raise ValueError("unknown type of linearization %s" % args.linearization)

# set weak form of convergence law for A
Ate = TestFunction(A)
Atr = TrialFunction(A)

n = FacetNormal(mesh)
vn = 0.5*(dot(sol_u, n) + abs(dot(sol_u, n)))

a_A = Atr*Ate*dx
L1 = dtc*(sol_A*inner(sol_u, nabla_grad(Ate))*dx
     - conditional(dot(sol_u, n) < 0, Ate*dot(sol_u, n)*1.0, 0.0)*ds
     - conditional(dot(sol_u, n) > 0, Ate*dot(sol_u, n)*sol_A, 0.0)*ds
     - (Ate('+') - Ate('-'))*(vn('+')*sol_A('+') - vn('-')*sol_A('-'))*dS)

sol_A1 = Function(A); sol_A2 = Function(A)
L2 = replace(L1, {sol_A: sol_A1}); L3 = replace(L1, {sol_A: sol_A2})    

probA1 = LinearVariationalProblem(a_A, L1, step_A)
solvA1 = LinearVariationalSolver(probA1)
probA2 = LinearVariationalProblem(a_A, L2, step_A)
solvA2 = LinearVariationalSolver(probA2)
probA3 = LinearVariationalProblem(a_A, L3, step_A)
solvA3 = LinearVariationalSolver(probA3)

# set weak form of convergence law for A
Hte = TestFunction(H)
Htr = TrialFunction(H)

a_H = Htr*Hte*dx
L1 = dtc*(sol_H*inner(sol_u, nabla_grad(Hte))*dx
     - conditional(dot(sol_u, n) < 0, Hte*dot(sol_u, n)*0.3/L, 0.0)*ds
     - conditional(dot(sol_u, n) > 0, Hte*dot(sol_u, n)*sol_H, 0.0)*ds
     - (Hte('+') - Hte('-'))*(vn('+')*sol_H('+') - vn('-')*sol_H('-'))*dS)

sol_H1 = Function(H); sol_H2 = Function(H)
L2 = replace(L1, {sol_H: sol_H1}); L3 = replace(L1, {sol_H: sol_H2})    

probH1 = LinearVariationalProblem(a_H, L1, step_H)
solvH1 = LinearVariationalSolver(probH1)
probH2 = LinearVariationalProblem(a_H, L2, step_H)
solvH2 = LinearVariationalSolver(probH2)
probH3 = LinearVariationalProblem(a_H, L3, step_H)
solvH3 = LinearVariationalSolver(probH3)


while t < Tfinal - 0.5*dt:
    WeakForm.update_va(mx, my, t, X, v_a, T, L)
    sol_uprevt.assign(sol_u)

	## output
    if ntstep % 160 == 0:
        if MONITOR_NL_ITER:
            PETSc.Sys.Print("[{0:2d}] Time: {1:>5.2e}; {2:>5.2e} days; nonlinear iter {3:>3d}".format(ntstep, t, t*1e3/60/60/24, nonlin_it_total))
        eta.interpolate(WeakForm.eta(sol_u, sol_A, sol_H, delta_min, Pstar))

        outfile_eta.write(eta, time=t)
        #outfile_va.write(v_a,  time=t)
        #outfile_u.write(sol_u, time=t)
        #outfile_A.write(sol_A, time=t)
        #outfile_H.write(sol_H, time=t)

    ### Advance A, H
    solvA1.solve()
    sol_A1.assign(sol_A + step_A)
    
    solvA2.solve()
    sol_A2.assign(0.75*sol_A + 0.25*(sol_A1 + step_A))
    
    solvA3.solve()
    sol_A.assign((1.0/3.0)*sol_A + (2.0/3.0)*(sol_A2 + step_A))

    solvH1.solve()
    sol_H1.assign(sol_H + step_H)
    
    solvH2.solve()
    sol_H2.assign(0.75*sol_H + 0.25*(sol_H1 + step_H))
    
    solvH3.solve()
    sol_H.assign((1.0/3.0)*sol_H + (2.0/3.0)*(sol_H2 + step_H))


    ### Solve the momentum equation
    if ntstep % 20 == 19:
        # initialize gradient
        g = assemble(grad, bcs=bcstep_u)
        g_norm_init = g_norm = norm(g)
        angle_grad_step_init = angle_grad_step = np.nan
        
        # initialize solver statistics
        lin_it       = 0
        lin_it_total = 0
        obj_val      = assemble(obj)
        step_length  = 0.0
        
        if args.linearization == 'stressvel':
            Vdmassweak = inner(TrialFunction(Vd), TestFunction(Vd)) * dx
            Md = assemble(Vdmassweak)
        if MONITOR_NL_ITER:
            PETSc.Sys.Print('{0:<3} "{1:>6}"{2:^20}{3:^14}{4:^15}{5:^15}{6:^10}'.format(
                  "Itn", "default", "Energy", "||g||_l2", 
                   "(grad,step)", "dual res", "step len"))

        Sresnorm = 0.0
        for itn in range(NL_SOLVER_MAXITER+1):
        #for itn in range(0):
            # print iteration line
            if MONITOR_NL_ITER:
                PETSc.Sys.Print("{0:>3d} {1:>6d}{2:>20.12e}{3:>14.6e}{4:>+15.6e}{5:>+15.6e}{6:>10f}".format(
                      itn, lin_it, obj_val, g_norm, angle_grad_step, Sresnorm, step_length))
        
            # stop if converged
            if (g_norm < 1e-13 or np.abs(angle_grad_step) < 1e-16) and itn > 1:
                nlsolve_success = True
                PETSc.Sys.Print("Stop reason: ||g|| too small; ||g|| reduction %3e." % float(g_norm/g_norm_init))
                break
            if g_norm < NL_SOLVER_GRAD_RTOL*g_norm_init:
                nlsolve_success = True
                PETSc.Sys.Print("Stop reason: Converged to rtol; ||g|| reduction %3e." % float(g_norm/g_norm_init))
                break
            if np.abs(angle_grad_step) < NL_SOLVER_GRAD_STEP_RTOL*np.abs(angle_grad_step_init):
                nlsolve_success = True
                PETSc.Sys.Print("Stop reason: Converged to rtol; (grad,step) reduction %3e." % \
                      np.abs(angle_grad_step/angle_grad_step_init))
                break
            # stop if step search failed
            if 0 < itn and not step_success:
                PETSc.Sys.Print("Stop reason: Step search reached maximum number of backtracking.")
                nlsolve_success = False
                break
        
            # set up the linearized system
            if args.linearization == 'stressvel':
                if 0 == itn:
                    Abstract.Vector.setZero(S)
                    Abstract.Vector.setZero(S_step)
                    Abstract.Vector.setZero(S_proj)
                else:
                    # project S to unit sphere
                    Sprojweak = WeakForm.hessian_dualUpdate_boundMaxMagnitude(S, Vd, 0.5)
                    b = assemble(Sprojweak)
                    solve(Md, S_proj.vector(), b)
                    Sresnorm = norm(assemble(dualres))
        
            # assemble linearized system
            problem = LinearVariationalProblem(hess, grad, step_u, bcs=bcstep_u)
            solver  = LinearVariationalSolver(problem, options_prefix="ns_")
            solver.solve()
            lin_it=solver.snes.ksp.getIterationNumber()
            lin_it_total += lin_it
        
            # solve dual variable step
            if args.linearization == 'stressvel':
                Abstract.Vector.scale(step_u, -1.0)
                b = assemble(dualStep)
                solve(Md, S_step.vector(), b)
                Abstract.Vector.scale(step_u, -1.0)
            
            # compute the norm of the gradient
            g = assemble(grad, bcs=bcstep_u)
            g_norm = norm(g)
        
            # check derivatives
            if NL_CHECK_GRADIENT and itn < 2:
                perturb_u       = Function(V)
                perturb_uscaled = Function(V)
        
                randrhs = Function(V)
                Abstract.Vector.setZero(randrhs)
                Abstract.Vector.addNoiseRandUniform(randrhs)
        
                # create random direction and apply the right bcs
                Abstract.Vector.setZero(perturb_u)
                Abstract.Vector.addNoiseRandUniform(perturb_u)
                WeakForm.applyBoundaryConditions(perturb_u, bcstep_u)
        
                # scale the direction so that perturb and sol have similar scale
                p = LinearVariationalProblem(Abstract.WeakForm.mass(V),
                                             Abstract.WeakForm.magnitude_scale(sol_u, perturb_u, V),
                                             perturb_uscaled, bcs=bcstep_u)
                s = LinearVariationalSolver(p, options_prefix="gradcheck_")
                s.solve()
                Abstract.CheckDerivatives.gradient(g.vector(), obj, sol_u, obj_perturb=perturb_uscaled, \
                        grad_perturb=perturb_uscaled, n_checks=8)
        
            # compute angle between step and (negative) gradient
            angle_grad_step = -step_u.vector().inner(g)
            if 0 == itn:
                angle_grad_step_init = angle_grad_step
        
            # initialize backtracking line search
            sol_uprev.assign(sol_u)
            step_length = 1.0
            step_success = False
        
            # run backtracking line search
            for j in range(NL_SOLVER_STEP_MAXITER):
                sol_u.vector().axpy(-step_length, step_u.vector())
                WeakForm.applyBoundaryConditions(sol_u, bcstep_u)
                obj_val_next = assemble(obj)
                if MONITOR_NL_STEPSEARCH and 0 < j:
                   PETSc.Sys.Print("Step search: {0:>2d}{1:>10f}{2:>20.12e}{3:>20.12e}".format(
                         j, step_length, obj_val_next, obj_val))
                if obj_val_next < obj_val + step_length*NL_SOLVER_STEP_ARMIJO*angle_grad_step:
                    if args.linearization == 'stressvel':
                        S.vector().axpy(step_length, S_step.vector())
                    obj_val = obj_val_next
                    step_success = True
                    break
                step_length *= 0.5
                sol_u.assign(sol_uprev)
            if not step_success:
                sol_u.assign(sol_uprev)
                #sol_u.vector().axpy(-step_length, step_u.vector())
                #obj_val = obj_val_next
                #step_success = True
            Abstract.Vector.scale(step_u, -step_length)
        
        PETSc.Sys.Print("%s: #iter %i, ||g|| reduction %3e, (grad,step) reduction %3e, #total linear iter %i." % \
            (
                args.linearization,
                itn,
                g_norm/g_norm_init,
                np.abs(angle_grad_step/angle_grad_step_init),
                lin_it_total
            )
        )
        nonlin_it_total += itn
        # stop if nonlinear solve failed
        if not nlsolve_success or itn == 100:
            PETSc.Sys.Print("[{0:2d}] Failed solving momentum equation".format(ntstep))
            break

    t += dt
    ntstep += 1

#======================================
# Output
#======================================

# output vtk file for solutions
if OUTPUT_VTK:
    File("vtk/"+args.linearization+"/solution_u.pvd").write(sol_u)
    File("vtk/"+args.linearization+"/solution_A.pvd").write(sol_A)
    File("vtk/"+args.linearization+"/solution_H.pvd").write(sol_H)
