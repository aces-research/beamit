import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
import sys
from beamit import Material
import copy

class Solver:

    def __init__(self, system):
        self.system = system
        # the left hand side matrix
        self.A = np.zeros([system.nequations, system.nequations])
        # the right hand side vector
        self.f = np.zeros([system.nequations, 1])
        # the solution (unknowns)
        self.solution = np.zeros([system.nequations, 1])
        # initialize the unknowns in the system
        self.initialize()
        # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
        self.bctypes = np.zeros([self.system.weak_form.function_space.N, self.system.weak_form.function_space.dof], dtype=np.int64)
        # boundary condition values matrix
        self.bcvalues = np.zeros([self.system.weak_form.function_space.N, self.system.weak_form.function_space.dof])

    # Function to initialize unknowns to the undeformed state of the beam
    def initialize(self):
        self.solution = np.reshape(self.system.state, [self.system.nequations, 1])
        pass
    
    # Function to set the boundary conditions types (Dirichlet and Neumann) and values
    # for each pair (node, dof) of the beam
    def set_boundary_conditions(self, bctypes, bcvalues):
        self.bctypes = copy.deepcopy(bctypes)
        self.bcvalues = copy.deepcopy(bcvalues)
        # loop on node
        for n in range(0, self.bcvalues.shape[0]):
            # loop on nodal degrees of freedom
            for d in range(0, self.bcvalues.shape[1]):
                # if Dirichlet bcs
                if (bctypes[n, d] == 1):
                    # shift the Dirichlet conditions to obtain the incremental boundary conditions
                    self.bcvalues[n, d] = bcvalues[n, d] - self.system.state[n, d]

    # Function to modify the boundary condition values
    def modify_boundary_condition_values(self, bcvalues):
        self.bcvalues = copy.deepcopy(bcvalues)
        # loop on node
        for n in range(0, self.bctypes.shape[0]):
            # loop on nodal degrees of freedom
            for d in range(0, self.bctypes.shape[1]):
                # if Dirichlet bcs
                if (self.bctypes[n, d] == 1):
                    # shift the Dirichlet conditions to obtain the incremental boundary conditions
                    self.bcvalues[n, d] = bcvalues[n, d] - self.system.state[n, d]

    # Function to reset the linear system
    def reset_system(self):
        self.A = np.zeros([self.system.nequations, self.system.nequations])
        self.f = np.zeros([self.system.nequations, 1])
        pass

    # Create arrays with entries being the global dof numbers of Dirichlet and Neumann dofs
    def create_dof_arrays(self):
        bctypes_vec = np.reshape(self.bctypes, [self.system.nequations, 1])
        global_dofs = np.arange(0, self.system.nequations, 1)
        Neumann_dofs = global_dofs[(bctypes_vec == 0).flatten()]
        Dirichlet_dofs = global_dofs[(bctypes_vec == 1).flatten()]
        return Dirichlet_dofs, Neumann_dofs
    
    # Function to apply the Dirichlet boundary conditions by static condensation
    def apply_static_condensation(self, Neumann_dofs):
        self.A = self.A[np.ix_(Neumann_dofs, Neumann_dofs)]
        self.f = self.f[Neumann_dofs]

    # Function to solve the linear system Ax = f using different methods
    def linear_system_solver(self, A, f, solver_type = None, precon_type = None, tol = 1.0E-06, maxiter = None):
        # the compressed sparse column version of A
        A_csc = csc_matrix(A, dtype=np.float64)
        # preconditioners
        if (precon_type == "aINV"): # approximate inverse
            A_z = lambda z: spla.spsolve(A_csc, z)
            precon = spla.LinearOperator(A.shape, A_z)
        elif (precon_type == "iLU"): # ILU
            A_iLU = spla.spilu(A_csc)
            precon = spla.LinearOperator(A.shape, A_iLU.solve)
        else: # Identity
            precon = None
        # solve the linear system
        if (solver_type == "spsolve"):
            x = spla.spsolve(A_csc, f).reshape([A.shape[1],1])
        elif (solver_type == "cg"):
            x = spla.cg(A_csc, f, tol=tol, maxiter=maxiter, M=precon)[0].reshape([A.shape[1],1])
        elif (solver_type == "bicg"):
            x = spla.bicg(A_csc, f, tol=tol, maxiter=maxiter, M=precon)[0].reshape([A.shape[1],1])
        elif (solver_type == "bicgstab"):
            x = spla.bicgstab(A_csc, f, tol=tol, maxiter=maxiter, M=precon)[0].reshape([A.shape[1],1])
        elif (solver_type == "gmres"):
            x = spla.gmres(A_csc, f, tol=tol, maxiter=maxiter, M=precon)[0].reshape([A.shape[1],1])
        else:
            x = np.linalg.solve(A, f)
        return x

class NewtonRaphsonSolver(Solver):

    def __init__(self, system):
        # invoke the parent (Solver) class
        Solver.__init__(self, system)
    
    def solve(self, Nmax = 10, tol = 1.0E-05, stop_factor = 1.0E02, LSsolver = None, LSprecon = None, LStol = 1.0E-06, LSmaxiter = None):
        # reset linear system before solving
        self.reset_system()
        # create the Dirichlet and Neumann global dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # If Dirichlet boundary conditions are not available
        if (Dirichlet_dofs.size == 0):
            sys.exit("\nDirichlet boundary conditions are not found.")
        else:
            # generate the nodal load vector
            nodal_loads = np.zeros([self.system.nequations, 1])
            nodal_loads[Neumann_dofs] += np.reshape(self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
            # generate the Dirichlet solution vector
            Dirichlet_solution = np.reshape(self.bcvalues, [self.system.nequations, 1])[Dirichlet_dofs]
        # Newton-Raphson iterations
        for i in range(0, Nmax):
            # assemble the linear system
            self.system.assemble(self.A, self.f, self.solution, nodal_loads = nodal_loads)
            # forces to be applied after static condensation of Dirichlet dofs
            if (i == 0): # in the first iteration
                static_condensation_forces = np.matmul(self.A[np.ix_(Neumann_dofs, Dirichlet_dofs)], Dirichlet_solution)
                # add the Dirichlet solution to the overall solution vector
                self.solution[Dirichlet_dofs] += Dirichlet_solution
            else: # after the first iteration
                static_condensation_forces = np.zeros([Neumann_dofs.shape[0], 1])
            # apply the Dirichlet BCs
            self.apply_static_condensation(Neumann_dofs)
            log_det_A = np.linalg.slogdet(self.A)[1]
            # checks in the first iteration
            if (i == 0):
                # not enough fixity in the system
                if ((log_det_A == np.inf) or (log_det_A == -np.inf)):
                    sys.exit("\nSystem is not fixed properly.")
                # report
                print("\nStarting the Newton-Raphson iterations!!!")
            # checks after the first iteration
            else:
                # instability in the system
                if ((log_det_A == np.inf) or (log_det_A == -np.inf)):
                    sys.exit("\nInstability encountered in the system.")
            # solve the linear system
            solution_increment = self.linear_system_solver(self.A, (self.f)-static_condensation_forces, solver_type=LSsolver, precon_type=LSprecon, tol=LStol, maxiter=LSmaxiter)
            # update the overall solution vector
            self.solution[Neumann_dofs] += solution_increment
            # assess convergence
            # the current residual (= f_ext - f_int)
            updated_residual = np.zeros([self.system.nequations, 1])
            self.system.assemble_residual(updated_residual, self.solution, nodal_loads, element_loads_info = None)
            # the residual norm at all the NEUMANN NODES
            res_L2_norm = np.linalg.norm(updated_residual[Neumann_dofs], ord=2)
            print("\nIteration:",i+1,", Residual L2-norm = %.2e" % (res_L2_norm))
            if (res_L2_norm <= tol):
                print("\nSolver converged!!!")
                break
            else:
                # reset the linear system
                self.reset_system()
        # stop the computations if the final residual norm is too high
        if (res_L2_norm >= tol*stop_factor):
            sys.exit("\nThe final residual norm is too high to proceed.")
        # update the system attributes
        self.system.update(self.solution)

class DynamicSolver(Solver):

    def __init__(self, system):
        # invoke the parent (Solver) class
        Solver.__init__(self, system)
        # initialize the mass and damping matrices
        self.M = np.zeros([system.nequations, system.nequations])
        self.C = np.zeros([system.nequations, system.nequations])
        # the "velocity" (linear velocities and time derivative of the tangents)
        self.velocity = np.zeros([system.nequations, 1])
        # the "acceleration" (linear accelerations and double time derivative of the tangents)
        self.acceleration = np.zeros([system.nequations, 1])

    # Function to reset the dynamic linear system
    def reset_system(self):
        super().reset_system()
        self.M = np.zeros([self.system.nequations, self.system.nequations])
        self.C = np.zeros([self.system.nequations, self.system.nequations])
        pass
    
    # Function to set the initial conditions (position and velocity) of the system
    def set_initial_conditions(self, initial_position, initial_velocity):
        self.solution = copy.deepcopy(np.reshape(initial_position, [self.system.nequations, 1]))
        self.velocity = copy.deepcopy(np.reshape(initial_velocity, [self.system.nequations, 1]))
        # compute initial accelerations
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        if (Dirichlet_dofs.size == 0):
            sys.exit("\nSet the boundary conditions before the initial conditions.")
        nodal_loads = np.zeros([self.system.nequations, 1])
        nodal_loads[Neumann_dofs] += np.reshape(self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
        self.reset_system()
        self.system.assemble_residual(self.f, self.solution, nodal_loads = nodal_loads, element_loads_info = None)
        self.system.assemble_mass(self.M, self.solution)
        self.acceleration = self.linear_system_solver(self.M, self.f)
        # update the system attributes
        self.system.update(self.solution)

class ImplicitNewmarkSolver(DynamicSolver):

    def __init__(self, system):
        # invoke the parent (DynamicSolver) class
        DynamicSolver.__init__(self, system)

    def solve(self, dt, beta = 0.25, gamma = 0.50, Nmax = 10, tol = 1.0E-05, stop_factor = 1.0E02, LSsolver = None, LSprecon = None, LStol = 1.0E-06, LSmaxiter = None):
        # constants in the time integration scheme
        c0 = 1.0/(beta*(dt**2.0))
        c1 = 1.0/(beta*dt)
        c2 = (1.0/(2.0*beta)) - 1.0
        c3 = (1.0-gamma)*dt
        c4 = gamma*dt
        c5 = (0.5-beta)*(dt**2.0)
        # reset the system before solving
        self.reset_system()
        # create the Dirichlet and Neumann global dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # If Dirichlet boundary conditions are not available
        if (Dirichlet_dofs.size == 0):
            sys.exit("\nDirichlet boundary conditions are not found.")
        else:
            # generate the nodal load vector
            nodal_loads = np.zeros([self.system.nequations, 1])
            nodal_loads[Neumann_dofs] += np.reshape(self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
            # generate the Dirichlet solution vector
            Dirichlet_solution = np.reshape(self.bcvalues, [self.system.nequations, 1])[Dirichlet_dofs]
            # update the solution, velocity and acceleration of Dirichlet Dofs
            # Assuming only displacements are applied at the Dirichlet boundaries!!!
            acceleration_prev_Dirichlet = copy.deepcopy(self.acceleration[Dirichlet_dofs])
            velocity_prev_Dirichlet = copy.deepcopy(self.velocity[Dirichlet_dofs])
            self.acceleration[Dirichlet_dofs] = (Dirichlet_solution - (dt*velocity_prev_Dirichlet) - (c5*acceleration_prev_Dirichlet))*c0
            self.velocity[Dirichlet_dofs] = velocity_prev_Dirichlet + (c3*acceleration_prev_Dirichlet) + (c4*self.acceleration[Dirichlet_dofs])
            self.solution[Dirichlet_dofs] += Dirichlet_solution
            # initialize total solution increment in the current step
            solution_step = np.zeros([Neumann_dofs.shape[0], 1])
            # velocity and acceleration of the Neumann Dofs from the previous step
            velocity_prev_Neumann = copy.deepcopy(self.velocity[Neumann_dofs])
            acceleration_prev_Neumann = copy.deepcopy(self.acceleration[Neumann_dofs])
        # Newton-Raphson iterations
        for i in range(0, Nmax):
            # assemble the stiffness matrix and force vector
            self.system.assemble(self.A, self.f, self.solution, nodal_loads = nodal_loads)
            # YET TO ADD ROTATIONAL INERTIA RELATED UPDATES IN THIS SOLVER!
            # mass matrix contribution to the left hand side matrix
            self.A += c0*self.M
            # inertial force contribution to the right hand side vector
            if (i == 0): # in the first iteration
                self.f += (c1*np.matmul(self.M, self.velocity)) + (c2*np.matmul(self.M, self.acceleration))
            else: # from the second iteration
                self.f -= np.matmul(self.M, self.acceleration)
            # apply the Dirichlet BCs
            self.apply_static_condensation(Neumann_dofs)
            log_det_A = np.linalg.slogdet(self.A)[1]
            # checks in the first iteration
            if (i == 0):
                # not enough fixity in the system
                if ((log_det_A == np.inf) or (log_det_A == -np.inf)):
                    sys.exit("\nSystem is not fixed properly.")
                # report
                print("\nStarting the Newton-Raphson iterations!!!")
            # checks after the first iteration
            else:
                # instability in the system
                if ((log_det_A == np.inf) or (log_det_A == -np.inf)):
                    sys.exit("\nInstability encountered in the system.")
            # solve the linear system
            solution_increment = self.linear_system_solver(self.A, self.f, solver_type=LSsolver, precon_type=LSprecon, tol=LStol, maxiter=LSmaxiter)
            # update the total solution increment in the current step
            solution_step += solution_increment
            # update the overall solution, velocity and acceleration vectors of the Neumann Dofs
            self.solution[Neumann_dofs] += solution_increment
            self.acceleration[Neumann_dofs] = (c0*solution_step) - (c1*velocity_prev_Neumann) - (c2*acceleration_prev_Neumann)
            self.velocity[Neumann_dofs] = velocity_prev_Neumann + (c3*acceleration_prev_Neumann) + (c4*self.acceleration[Neumann_dofs])
            # assess convergence
            # the current residual (= f_external - f_internal - f_inertial)
            updated_residual = np.zeros([self.system.nequations, 1])
            self.system.assemble_residual(updated_residual, self.solution, nodal_loads, element_loads_info = None)
            updated_residual -= np.matmul(self.M, self.acceleration)
            # the residual norm at all the NEUMANN NODES
            res_L2_norm = np.linalg.norm(updated_residual[Neumann_dofs], ord=2)
            print("\nIteration:",i+1,", Residual L2-norm = %.2e" % (res_L2_norm))
            if (res_L2_norm <= tol):
                print("\nSolver converged!!!")
                break
            else:
                # reset the linear system
                self.reset_system()
        # stop the computations if the final residual norm is too high
        if (res_L2_norm >= tol*stop_factor):
            sys.exit("\nThe final residual norm is too high to proceed.")
        # update the system attributes
        self.system.update(self.solution)

class ExplicitNewmarkSolver(DynamicSolver):

    def __init__(self, system):
        # invoke the parent (Solver) class
        DynamicSolver.__init__(self, system)
        # initialize the stable time step size
        self.stable_time_step = None
        # compute the lumped mass
        self.system.assemble_mass(self.M, self.solution)
    
    # Function to compute the natural frequencies of the system
    def compute_system_frequencies(self):
        # create the Dirichlet and Neumann global dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # if Dirichlet boundary conditions are not available
        if (Dirichlet_dofs.size == 0):
            sys.exit("\nDirichlet boundary conditions are not found.")
        stiffness = np.zeros([self.system.nequations, self.system.nequations])
        residual = np.zeros([self.system.nequations, 1])
        self.system.assemble(stiffness, residual, self.solution, nodal_loads = np.zeros([self.system.nequations, 1]))
        eig_vals, _ = sp.linalg.eig(stiffness[np.ix_(Neumann_dofs, Neumann_dofs)], self.M[np.ix_(Neumann_dofs, Neumann_dofs)])
        # compute the complex valued Eigen frequencies of the system
        eig_freqs = np.sqrt(eig_vals)
        if (not (((eig_freqs.real > 0.0).all()) and ((eig_freqs.imag >= 0.0).all()))):
            print("\nEither the real and/or imaginary parts of the Eigen frequencies are negative.")
            print("\nThe non-positive real frequencies:", eig_freqs.real[eig_freqs.real <= 0.0])
            print("\nThe negative imaginary frequencies:", eig_freqs.imag[eig_freqs.imag < 0.0])
            print("\nThe real parts of negative imaginary frequencies:", eig_freqs.real[eig_freqs.imag < 0.0])
            print("\nThe real part of fundamental Eigen frequency:", eig_freqs.real[(np.absolute(eig_freqs)).argmin()])
            sys.exit()
        return np.absolute(eig_freqs)
    
    # Function to compute and set the stable time step
    # probably should consider degrading modulus in case of damage!!!
    def set_stable_time_step(self, time_factor = 0.90):
        print("\nRunning stable time computations!!!")
        sys_freqs = self.compute_system_frequencies()
        # get the maximum frequency of the system
        max_sys_freq = sys_freqs[sys_freqs.argmax()]
        self.stable_time_step = time_factor*(2.0/(max_sys_freq.real))
    
    # Function to set the boundary conditions and the stable time step
    def set_boundary_conditions(self, bctypes, bcvalues):
        super().set_boundary_conditions(bctypes, bcvalues)
        self.set_stable_time_step()

    def solve(self, dt = None, LSsolver = None, LSprecon = None, LStol = 1.0E-06, LSmaxiter = None):
        # if the time step size input is not provided
        if (dt == None):
            dt = self.stable_time_step
        # perform stability check
        elif (dt > self.stable_time_step):
            sys.exit("\nThe chosen time step size makes the solver unstable in time.")
        # reset the system before solving
        Solver.reset_system()
        # create the Dirichlet and Neumann global dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # generate the nodal load vector
        nodal_loads = np.zeros([self.system.nequations, 1])
        nodal_loads[Neumann_dofs] += np.reshape(self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
        # generate the Dirichlet solution vector
        Dirichlet_solution = np.reshape(self.bcvalues, [self.system.nequations, 1])[Dirichlet_dofs]
        # Assuming only displacements are applied at the Dirichlet boundaries!!!
        # the PREDICTOR
        solution_prev_Dirichlet = copy.deepcopy(self.solution[Dirichlet_dofs])
        velocity_prev_Dirichlet = copy.deepcopy(self.velocity[Dirichlet_dofs])
        # for the Dirichlet DoFs
        self.solution[Dirichlet_dofs] += Dirichlet_solution
        self.velocity[Dirichlet_dofs] = (self.solution[Dirichlet_dofs] - solution_prev_Dirichlet)/dt
        self.acceleration[Dirichlet_dofs] = (self.velocity[Dirichlet_dofs] - velocity_prev_Dirichlet)/dt
        # for the Neumann DoFs
        self.solution[Neumann_dofs] += (dt*self.velocity[Neumann_dofs]) + (((dt**2.0)/2.0)*self.acceleration[Neumann_dofs])
        self.velocity[Neumann_dofs] += ((dt/2.0)*self.acceleration[Neumann_dofs])
        # assemble the residual
        if (isinstance(self.system.weak_form.material, (Material.CohesiveInterfaceMaterial))):
            self.system.assemble_residual(self.f, self.solution, nodal_loads, element_loads_info = None, update_internal = True)
        else:
            self.system.assemble_residual(self.f, self.solution, nodal_loads, element_loads_info = None)
        # the CORRECTOR
        # solve the semi-discrete SOE for accelerations of the Neumann Dofs
        self.acceleration[Neumann_dofs] = self.f[Neumann_dofs] / np.diag(self.M)[Neumann_dofs]
        self.velocity[Neumann_dofs] += ((dt/2.0)*self.acceleration[Neumann_dofs])
        # update the system attributes
        self.system.update(self.solution)