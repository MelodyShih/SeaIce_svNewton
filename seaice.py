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
    outfile_va    = File("/scratch/vtk/"+args.linearization+"/sol_va_ts.pvd")
    outfile_eta   = File("/scratch/vtk/"+args.linearization+"/sol_eta_ts_"+str(L)+".pvd")
    outfile_shear = File("/scratch/vtk/"+args.linearization+"/sol_shear_ts_"+str(L)+".pvd")
    outfile_A     = File("/scratch/vtk/"+args.linearization+"/sol_A_ts_"+str(L)+".pvd")
    outfile_H     = File("/scratch/vtk/"+args.linearization+"/sol_H_ts_"+str(L)+".pvd")
    outfile_u     = File("/scratch/vtk/"+args.linearization+"/sol_u_ts_"+str(L)+".pvd")

#outfile_e   = File("/scratch/vtk/"+args.linearization+"/delta_"+str(L)+".pvd")


## Domain: (0,1)x(0,1)
distp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
dim = 2
Nx = Ny = int(512/4.0)
Lx = Ly = 512*1000/L
basemesh = RectangleMesh(Nx, Ny, Lx, Ly, distribution_parameters=distp, quadrilateral=False)
nref = 1
mh = MeshHierarchy(basemesh, nref, reorder=True,
                   distribution_parameters=distp)
mesh = mh[-1]
PETSc.Sys.Print("[info] dx in km", (512*1000/L)/Nx/2**nref)
PETSc.Sys.Print("[info] Nx", Nx)

Tfinal = 2.1*24*60*60/T #2/T #days
dt = 1.8 # 1.8/2 for N=128, 1.8/4 for N=256
dtc = Constant(dt)
t   = 0.0
ntstep = 0
PETSc.Sys.Print("[info] Tfinal in days", Tfinal*T/24/60/60)
PETSc.Sys.Print("[info] dt in days", dt*T/24/60/60)

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
#sol_u.interpolate(as_vector([(2*X[1]-Lx)/Lx,-(2*X[0]-Ly)/Ly]))
sol_uprevt.assign(sol_u)

# A_0 = 1.0
#bell_r0 = 32; bell_x0 = 256; bell_y0 = 256+96
#bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(X[0]-bell_x0, 2) + pow(X[1]-bell_y0, 2))/bell_r0, 1.0)))
#sol_A.interpolate(1+bell)
sol_A.interpolate(Constant(1.0))

# H_0
#sol_H.interpolate((0.3/L + 0.005/L*(sin(60*X[0]/1e6*L) + sin(30*X[1]/1e6*L)))/G)
sol_H.interpolate((0.3 + 0.005*(sin(60*X[0]/1e6*L) + sin(30*X[1]/1e6*L)))/G)
#Abstract.Vector.setValue(sol_H, 0.3/G)

# v_ocean
v_ocean_max = 0.01*T/L #m/s
v_ocean.interpolate(as_vector([v_ocean_max*(2*X[1]-Lx)/Lx,-v_ocean_max*(2*X[0]-Ly)/Ly]))

# v_a
mx = Constant(0.0)
my = Constant(0.0)
WeakForm.update_va(mx, my, 0.0, X, v_a, T, L)

# visualize initial value
if OUTPUT_VTK_INIT:
    File("/scratch/vtk/"+args.linearization+"/initial_vo_"+str(L)+".pvd").write(v_ocean)
    File("/scratch/vtk/"+args.linearization+"/initial_va_"+str(L)+".pvd").write(v_a)
    File("/scratch/vtk/"+args.linearization+"/initial_H_"+str(L)+".pvd").write(sol_H)

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
else:
    raise ValueError("unknown type of linearization %s" % args.linearization)

transfer = firedrake.TransferManager()

# set weak form of convergence law for A
Ate = TestFunction(A)
Atr = TrialFunction(A)

n = FacetNormal(mesh)
vn = 0.5*(dot(sol_u, n) + abs(dot(sol_u, n)))

a_A = Atr*Ate*dx
L1 = dtc*(
     sol_A*inner(sol_u, nabla_grad(Ate))*dx
     #- conditional(dot(sol_u, n) < 0, Ate*dot(sol_u, n)*1.0, 0.0)*ds
     #- conditional(dot(sol_u, n) > 0, Ate*dot(sol_u, n)*sol_A, 0.0)*ds
     - (Ate('+') - Ate('-'))*(vn('+')*sol_A('+') - vn('-')*sol_A('-'))*dS)

sol_A1 = Function(A); sol_A2 = Function(A)
L2 = replace(L1, {sol_A: sol_A1}); L3 = replace(L1, {sol_A: sol_A2})    

probA1 = LinearVariationalProblem(a_A, L1, step_A)
solvA1 = LinearVariationalSolver(probA1)
solvA1.set_transfer_manager(transfer)
probA2 = LinearVariationalProblem(a_A, L2, step_A)
solvA2 = LinearVariationalSolver(probA2)
probA3 = LinearVariationalProblem(a_A, L3, step_A)
solvA3 = LinearVariationalSolver(probA3)

# set weak form of convergence law for A
Hte = TestFunction(H)
Htr = TrialFunction(H)

a_H = Htr*Hte*dx
L1 = dtc*(
     sol_H*inner(sol_u, nabla_grad(Hte))*dx
     #- conditional(dot(sol_u, n) < 0, Hte*dot(sol_u, n)*0.3/L, 0.0)*ds
     #- conditional(dot(sol_u, n) > 0, Hte*dot(sol_u, n)*sol_H, 0.0)*ds
     - (Hte('+') - Hte('-'))*(vn('+')*sol_H('+') - vn('-')*sol_H('-'))*dS)

sol_H1 = Function(H); sol_H2 = Function(H)
L2 = replace(L1, {sol_H: sol_H1}); L3 = replace(L1, {sol_H: sol_H2})    

probH1 = LinearVariationalProblem(a_H, L1, step_H)
solvH1 = LinearVariationalSolver(probH1)
solvH1.set_transfer_manager(transfer)
probH2 = LinearVariationalProblem(a_H, L2, step_H)
solvH2 = LinearVariationalSolver(probH2)
probH3 = LinearVariationalProblem(a_H, L3, step_H)
solvH3 = LinearVariationalSolver(probH3)

if args.linearization == 'stressvel':
    Vdmassweak = inner(TrialFunction(Vd), TestFunction(Vd)) * dx
    #Md = assemble(Vdmassweak)
    Mdinv = assemble(Tensor(Vdmassweak).inv).petscmat

Amassweak = inner(TrialFunction(A), TestFunction(A)) * dx
Ainv = assemble(Tensor(Amassweak).inv).petscmat

while t < Tfinal - 0.5*dt and ntstep == 0:
    WeakForm.update_va(mx, my, t, X, v_a, T, L)
    sol_uprevt.assign(sol_u)

	## output
    if ntstep % 1 == 0:
        if MONITOR_NL_ITER:
            PETSc.Sys.Print("[{0:2d}] Time: {1:>5.2e}; {2:>5.2e} days; nonlinear iter {3:>3d}".format(ntstep, t, t*1e3/60/60/24, nonlin_it_total))
            energy  = assemble(0.5*rhoice*900*sol_H*inner(sol_u/T*L, sol_u/T*L)*dx)
            E = sym(nabla_grad(sol_u))
            meandiv = assemble(abs(tr(E))*dx)
            shearweak = 2*sqrt(-det(dev(E)))
            meanshear = assemble(shearweak*dx)
            meanspeed = assemble(sqrt(inner(sol_u, sol_u))*dx)
            meandeform = assemble(sqrt(tr(E)*tr(E) + shearweak*shearweak)*dx)
            PETSc.Sys.Print("[{0:2d}] Statistic: {1:>8.4e} {2:>8.4e} {3:>8.4e} {4:>8.4e} {5:>8.4e}".format(ntstep,energy,meandiv,meanshear,meanspeed,meandeform))
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
        if OUTPUT_YC:
            sigmaI.interpolate(WeakForm.sigmaI(sol_u, sol_A, sol_H, delta_min, Pstar))
            sigmaII.interpolate(WeakForm.sigmaII(sol_u, sol_A, sol_H, delta_min, Pstar))
            fig, ax = plt.subplots(figsize=(10,5))
            ax.scatter(sigmaI.vector()[:], sigmaII.vector()[:], s=0.1)
            ax.set_xlim(-1,0)
            x = np.linspace(-1,0,1000)
            y = np.sqrt(1./4 - (x + 0.5)**2)/2
            ax.plot(x,y, 'r-')
            plt.savefig("stress_"+str(ntstep)+".png")
            #plt.show()

    ### Advance A, H
    solvA1.solve()
    sol_A1.assign(sol_A + step_A)
    
    #solvA2.solve()
    #sol_A2.assign(0.75*sol_A + 0.25*(sol_A1 + step_A))
    
    #solvA3.solve()
    #sol_A.assign((1.0/3.0)*sol_A + (2.0/3.0)*(sol_A2 + step_A))
    sol_A.assign(sol_A1)
    Aprojweak, _ = WeakForm.hessian_dualUpdate_boundMaxMagnitude(sol_A, A, 1.0)
    with assemble(Aprojweak).dat.vec_ro as v:
        with sol_A.dat.vec as aproj:
            Ainv.mult(v, aproj)
    Ate = TestFunction(A)
    Abdweak = conditional(lt(sol_A, 0.0), 0.0, sol_A)*Ate*dx
    with assemble(Abdweak).dat.vec_ro as v:
        with sol_A.dat.vec as aproj:
            Ainv.mult(v, aproj)
    

    solvH1.solve()
    sol_H1.assign(sol_H + step_H)
    #
    #solvH2.solve()
    #sol_H2.assign(0.75*sol_H + 0.25*(sol_H1 + step_H))
    #
    #solvH3.solve()
    #sol_H.assign((1.0/3.0)*sol_H + (2.0/3.0)*(sol_H2 + step_H))
    sol_H.assign(sol_H1)

    Hte = TestFunction(H)
    Hbdweak = conditional(lt(sol_H, 0.0), 0.0, sol_H)*Hte*dx
    with assemble(Hbdweak).dat.vec_ro as v:
        with sol_H.dat.vec as hproj:
            Ainv.mult(v, hproj)

    ### Solve the momentum equation
    if ntstep % 1 == 0:
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
            #if (g_norm < 1e-13 or np.abs(angle_grad_step) < 1e-16) and itn > 1:
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
                    #b = assemble(Sprojweak)
                    #solve(Md, S_proj.vector(), b)
                    Sresnorm     = norm(assemble(dualres))
                    Sprojpercent = assemble(S_ind)/Lx/Ly
        
            # assemble linearized system
            mg_levels_solver_patch = {
                #"ksp_monitor_true_residual": None,
                "ksp_type": "fgmres",
                "ksp_norm_type": "unpreconditioned",
                "ksp_max_it": 5,
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_construct_dim": 0,
                "pc_star_backend": "petscasm",
                "pc_star_sub_sub_pc_factor_in_place": None,
            }
            mg_levels_solver = {
                "ksp_type": "fgmres",
                "ksp_max_it": 10,
                "ksp_norm_type": "unpreconditioned",
                #"ksp_view": None,
                #"ksp_monitor_true_residual": None,
                #"pc_type": "bjacobi",
            }
            params = {
                "snes_type": "ksponly",
                "snes_rtol": 1e-4,
                "snes_atol": 1e-10,
                #"mat_type": "aij",
                #"pmat_type": "aij",
                "ksp_rtol": 1.0e-4,
                "ksp_atol": 1.0e-10,
                "ksp_max_it": 500,
                #"ksp_monitor_true_residual": None,
                #"ksp_converged_reason": None,
                #"ksp_type": "preonly",
                #"pc_type": "lu",
                #"pc_factor_mat_solver_type": "mumps",
                #"ksp_view": None,
                "ksp_type": "fgmres",
                "ksp_gmres_restart": 200,
                "pc_type": "mg",
                #"pc_mg_type": "full",
                #"pc_mg_type": "multiplicative",
                "pc_mg_cycle_type": "v",
                "mg_levels": mg_levels_solver,
                "mg_coarse_pc_type": "python",
                "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                "mg_coarse_assembled_pc_type": "lu",
                "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
            }
            problem = LinearVariationalProblem(hess, grad, step_u, bcs=bcstep_u)
            solver  = LinearVariationalSolver(problem, solver_parameters=params,
                                              options_prefix="ns_")
            solver.set_transfer_manager(transfer)
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
        
        nonlin_it_total += itn
        # stop if nonlinear solve failed
        if not nlsolve_success or itn == NL_SOLVER_MAXITER:
            PETSc.Sys.Print("[{0:2d}] Failed solving momentum equation".format(ntstep))
            break

    t += dt
    ntstep += 1

#======================================
# Output
#======================================

# output vtk file for solutions
if OUTPUT_VTK:
    File("/scratch/vtk/"+args.linearization+"/solution_u.pvd").write(sol_u)
    File("/scratch/vtk/"+args.linearization+"/solution_A.pvd").write(sol_A)
    File("/scratch/vtk/"+args.linearization+"/solution_H.pvd").write(sol_H)
