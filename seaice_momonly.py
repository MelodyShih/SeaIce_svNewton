from firedrake import *
from firedrake.petsc import PETSc
import math
import WeakForm
import Abstract.Vector
import Abstract.WeakForm
import Abstract.CheckDerivatives

import argparse
import numpy as np
import matplotlib.pyplot as plt
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
NL_SOLVER_MAXITER = 500
NL_SOLVER_STEP_MAXITER = 15
NL_SOLVER_STEP_ARMIJO = 1.0e-4
nonlin_it_total = 0

## Scaling
T = 1e3       # 1e3 second
L = 1e3       # 1e3 m
G = 1         # 1 m

## Output
OUTPUT_VTK = False
OUTPUT_VTK_INIT = False
OUTPUT_YC = False
if OUTPUT_VTK:
    outfile_va    = File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/sol_va_ts.pvd")
    outfile_eta   = File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/sol_eta_ts_"+str(L)+".pvd")
    outfile_shear = File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/sol_shear_ts_"+str(L)+".pvd")
    outfile_A     = File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/sol_A_ts_"+str(L)+".pvd")
    outfile_H     = File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/sol_H_ts_"+str(L)+".pvd")
    outfile_u     = File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/sol_u_ts_"+str(L)+".pvd")

#outfile_e   = File("/scratch/vtk/"+args.linearization+"/delta_"+str(L)+".pvd")


## Domain: (0,1)x(0,1)
dim = 2
Nx = Ny = int(512/1.0)
Lx = Ly = 512*1000/L
mesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False)
PETSc.Sys.Print("[info] dx in km", (512*1000/L)/Nx)
PETSc.Sys.Print("[info] Nx", Nx)

dt = 1.8 # 1.8/2 for N=128, 1.8/4 for N=256
dtc = Constant(dt)
t   = 0.0
ntstep = 0
PETSc.Sys.Print("[info] dt in days", dt*T/24/60/60)
PETSc.Sys.Print("[info] dt in minutes", dt*T/60)

## Discretization: 
# v: sea ice velocity
# A: sea ice concentration
# H: mean sea ice thickness
velt  = FiniteElement("CG", mesh.ufl_cell(), 1)
vdelt = TensorElement("DG", mesh.ufl_cell(), 2)
Aelt  = FiniteElement("DG", mesh.ufl_cell(), 0)
Helt  = FiniteElement("DG", mesh.ufl_cell(), 0)

V = VectorFunctionSpace(mesh, velt)
Vd= FunctionSpace(mesh, vdelt)
Vd1 = FunctionSpace(mesh, "DG", 0)
A = FunctionSpace(mesh, Aelt)
H = FunctionSpace(mesh, Helt)

## Parameteres
# fc: Coriolis
fc = 0.0 #1.46e-4/T #s^{-1}
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
delta_min = 2e-9*T #2e-9/1e3/T newton failed when line search; stressvel works
PETSc.Sys.Print("[info] delta_min", delta_min)

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
shear      = Function(Vd1)
strainrate = Function(Vd)
delta      = Function(Vd1)
if OUTPUT_YC:
    sigmaI  = Function(FunctionSpace(mesh, velt))
    sigmaII = Function(FunctionSpace(mesh, velt))

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
sol_uprevt.assign(sol_u)

# A_0 = 1.0
r = 0.04-(X[0]/1000-0.25)**2-(X[1]/1000-0.25)**2  
r1 = (2*X[0]/1000)**2-(2*X[1]/1000)+0.1 
sol_A.interpolate(1-0.5*exp(-800*abs(r))-0.4*exp(-90*abs(r1))-0.4*exp(-90*abs(r1+0.7)))

# H_0
sol_H.interpolate(2.0*(1-0.5*exp(-800*abs(r))-0.4*exp(-90*abs(r1))-0.4*exp(-90*abs(r1+0.7))))

# v_ocean
v_ocean.interpolate(as_vector([Constant(0.0),Constant(0.0)]))

# v_a
v_a.interpolate(as_vector([Constant(5.0*T/L),Constant(5.0*T/L)]))

# visualize initial value
if OUTPUT_VTK_INIT:
    File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/initial_vo_"+str(L)+".pvd").write(v_ocean)
    File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/initial_va_"+str(L)+".pvd").write(v_a)
    File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/initial_H_"+str(L)+".pvd").write(sol_H)
    File("/scratch1/04841/tg841407/vtk_momonly/"+args.linearization+"/initial_A_"+str(L)+".pvd").write(sol_A)

## Weak Form
# set weak forms of objective functional and gradient
obj  = WeakForm.objective(sol_u, sol_uprevt, sol_A, sol_H, V, rhoice, dt, Ca, rhoa, v_a, \
                         Co, rhoo, v_ocean, delta_min, Pstar, fc)
grad = WeakForm.gradient(sol_u, sol_uprevt, sol_A, sol_H, V, rhoice, dt, Ca, rhoa, v_a, \
                         Co, rhoo, v_ocean, delta_min, Pstar, fc)

# set weak form of Hessian and forms related to the linearization
if args.linearization == 'stdnewton':
    hess = WeakForm.hessian_NewtonStandard(sol_u, sol_A, sol_H, V, rhoice, \
               dt, Ca, rhoa, v_a, Co, rhoo, v_ocean, delta_min, Pstar, fc)
    hess_1 = WeakForm.hessian_NewtonStandard(sol_u, sol_A, sol_H, V, rhoice, \
               dt, Ca, rhoa, v_a, 0.0, rhoo, v_ocean, delta_min, Pstar, fc)
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
               dt, Ca, rhoa, v_a, Co, rhoo, v_ocean, delta_min, Pstar, fc)
    hess_1 = WeakForm.hessian_NewtonStressvel(sol_u, S_proj, sol_A, sol_H, V, rhoice, \
               dt, Ca, rhoa, v_a, 0.0, rhoo, v_ocean, delta_min, Pstar, fc)
else:
    raise ValueError("unknown type of linearization %s" % args.linearization)

if args.linearization == 'stressvel':
    Vdmassweak = inner(TrialFunction(Vd), TestFunction(Vd))*dx
    Mdinv = assemble(Tensor(Vdmassweak).inv).petscmat

Amassweak = inner(TrialFunction(A), TestFunction(A))*dx
Ainv = assemble(Tensor(Amassweak).inv).petscmat

### Solve the momentum equation
# initialize gradient
g = assemble(grad, bcs=bcstep_u)
g_norm_init = g_norm = norm(g)
angle_grad_step_init = angle_grad_step = np.nan

# initialize solver statistics
lin_it       = 0
lin_it_total = 0
obj_val      = assemble(obj)
step_length  = 0.0

if MONITOR_NL_ITER:
    PETSc.Sys.Print('{0:<3} "{1:>6}"{2:^20}{3:^14}{4:^15}{5:^15}{6:^10}{7:^10}'.format(
          "Itn", "default", "Energy", "||g||_l2", 
           "(grad,step)", "dual res", "dual(%)", "step len"))

Sresnorm = 0.0
Sprojpercent = 0.0

for itn in range(NL_SOLVER_MAXITER+1):
    # print iteration line
    if MONITOR_NL_ITER:
        PETSc.Sys.Print("{0:>3d} {1:>6d}{2:>20.12e}{3:>14.6e}{4:>+15.6e}{5:>15.6e}{6:>10.2f}{7:>10f}".format(
              itn, lin_it, obj_val, g_norm, angle_grad_step, Sresnorm, Sprojpercent*100, step_length))

    # stop if converged
    if (g_norm < 1e-13) and itn > 1:
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
        if 0 == itn and nonlin_it_total == 0:
            Abstract.Vector.setZero(S)
            Abstract.Vector.setZero(S_step)
            Abstract.Vector.setZero(S_proj)
        else:
            # project S to unit sphere
            Sprojweak, S_ind = WeakForm.hessian_dualUpdate_boundMaxMagnitude(S, Vd, sqrt(0.5))
            with assemble(Sprojweak).dat.vec_ro as v:
                with S_proj.dat.vec as sproj:
                    Mdinv.mult(v, sproj)
            Sresnorm     = norm(assemble(dualres))
            Sprojpercent = assemble(S_ind)/Lx/Ly

    # assemble linearized system
    params = {
        "snes_type": "ksponly",
        #"snes_monitor": None,
        "snes_atol": 1e-6,
        "snes_rtol": 1e-10,
        "mat_type": "aij",
        "pmat_type": "aij",
        "ksp_type": "preonly",
        "ksp_rtol": 1.0e-6,
        "ksp_atol": 1.0e-10,
        "ksp_max_it": 200,
        #"ksp_monitor_true_residual": None,
        #"ksp_converged_reason": None,
        "pc_type": "lu",
    }
    if itn == 0:
        problem = LinearVariationalProblem(hess_1, grad, step_u, bcs=bcstep_u)
    else:
        problem = LinearVariationalProblem(hess, grad, step_u, bcs=bcstep_u)
    solver  = LinearVariationalSolver(problem, solver_parameters=params,
                                      options_prefix="ns_")
    solver.solve()
    lin_it=solver.snes.ksp.getIterationNumber()
    lin_it_total += lin_it

    # solve dual variable step
    if args.linearization == 'stressvel':
        Abstract.Vector.scale(step_u, -1.0)
        b = assemble(dualStep)
        #solve(Md, S_step.vector(), b)
        with assemble(dualStep).dat.vec_ro as v:
            with S_step.dat.vec as sstep:
                Mdinv.mult(v, sstep)
        Abstract.Vector.scale(step_u, -1.0)
    
    # compute the norm of the gradient
    g = assemble(grad, bcs=bcstep_u)
    g_norm = norm(g)

    # check derivatives
    if NL_CHECK_GRADIENT:
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
        #step_length = 1.0
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



#======================================
# Output
#======================================

# output vtk file for solutions
if OUTPUT_VTK:
    eta.interpolate(WeakForm.eta(sol_u, sol_A, sol_H, delta_min, 27.5e3))
    E = sym(nabla_grad(sol_u))
    meandiv = assemble(abs(tr(E))*dx)
    shearweak = 2*sqrt(-det(dev(E)))
    shear.interpolate(shearweak)
    
    outfile_eta.write(eta, time=t*T/24/60/60)
    outfile_shear.write(shear, time=t*T/24/60/60)
    outfile_va.write(v_a,  time=t*T/24/60/60)
    outfile_u.write(sol_u, time=t*T/24/60/60)
    outfile_A.write(sol_A, time=t*T/24/60/60)
    outfile_H.write(sol_H, time=t*T/24/60/60)
