from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
import sys
import copy
from beamit.WeakForm.Utils import SolutionUpdateType
from beamit.WeakForm.ShearFlexibleGeometricallyExactWeakForm import \
    ShearFlexibleGeometricallyExactWeakFormCG, ShearFlexibleGeometricallyExactWeakFormDG

class Solver(ABC):

    def __init__(self, system):
        """
        Initialize the solver.

        Parameters:
            system: The system containing the weak form, state, and internal forces.
        """
        self.system = system
        # the left hand side matrix
        self.A = np.zeros([system.nequations, system.nequations])
        # the right hand side vector
        self.f = np.zeros([system.nequations, 1])
        # the solution (unknowns)
        self.solution = np.zeros([system.nequations, 1])
        # initialize the unknowns in the system
        self.solution = np.reshape(
            self.system.state, [self.system.nequations, 1])
        # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
        self.bctypes = np.zeros([self.system.weak_form.function_space.N,
                                self.system.weak_form.function_space.dof], dtype=np.int64)
        # boundary condition values matrix
        self.bcvalues = np.zeros(
            [self.system.weak_form.function_space.N, self.system.weak_form.function_space.dof])
        # translational and rotational dof indices
        dof = self.system.weak_form.function_space.dof
        dofspel = self.system.weak_form.function_space.npel * dof
        if (self.system.weak_form.function_space.discretization_type == "CG"):
            # number of translational and rotational dofs per node
            num_tns_dofs = (int)(self.system.weak_form.function_space.local_translational_dofs.size /
                                 self.system.weak_form.function_space.npel)
            num_rot_dofs = (int)(self.system.weak_form.function_space.local_rotational_dofs.size /
                                 self.system.weak_form.function_space.npel)
            self.translational_dof_indices = np.concatenate(
                [self.system.weak_form.function_space.local_translational_dofs[0:num_tns_dofs] +
                dof*i for i in range(self.system.weak_form.function_space.N)])
            self.rotational_dof_indices = np.concatenate(
                [self.system.weak_form.function_space.local_rotational_dofs[0:num_rot_dofs] +
                dof*i for i in range(self.system.weak_form.function_space.N)])
        elif (self.system.weak_form.function_space.discretization_type == "DG"):
            self.translational_dof_indices = np.concatenate(
                [self.system.weak_form.function_space.local_translational_dofs + 
                dofspel*i for i in range(self.system.weak_form.function_space.E)])
            self.rotational_dof_indices = np.concatenate(
                [self.system.weak_form.function_space.local_rotational_dofs + 
                dofspel*i for i in range(self.system.weak_form.function_space.E)])

    def set_boundary_conditions(self, bctypes, bcvalues):
        """
        Set the boundary conditions for the system.

        Parameters:
            bctypes: The boundary condition types (0 = Neumann, 1 = Dirichlet).
            bcvalues: The boundary condition values.
        """
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

    def modify_boundary_condition_values(self, bcvalues):
        """
        Modify the boundary condition values for the system.

        Parameters:
            bcvalues: The new boundary condition values.
        """
        self.bcvalues = copy.deepcopy(bcvalues)
        # loop on node
        for n in range(0, self.bctypes.shape[0]):
            # loop on nodal degrees of freedom
            for d in range(0, self.bctypes.shape[1]):
                # if Dirichlet bcs
                if (self.bctypes[n, d] == 1):
                    # shift the Dirichlet conditions to obtain the incremental boundary conditions
                    self.bcvalues[n, d] = bcvalues[n, d] - self.system.state[n, d]

    def reset_system(self):
        """
        Reset the linear system matrices and vectors.
        """
        self.A.fill(0.0)
        self.f.fill(0.0)

    def create_dof_arrays(self):
        """
        Create the Dirichlet and Neumann global dof arrays.

        Returns:
            Dirichlet_dofs: The global dof indices for Dirichlet boundary conditions.
            Neumann_dofs: The global dof indices for Neumann boundary conditions.
        """
        bctypes_vec = np.reshape(self.bctypes, [self.system.nequations, 1])
        global_dofs = np.arange(0, self.system.nequations, 1)
        Neumann_dofs = global_dofs[(bctypes_vec == 0).flatten()]
        Dirichlet_dofs = global_dofs[(bctypes_vec == 1).flatten()]
        return Dirichlet_dofs, Neumann_dofs

    def apply_static_condensation(self, Neumann_dofs):
        """
        Apply the Dirichlet boundary conditions by static condensation.

        Parameters:
            Neumann_dofs: The global dof indices for Neumann boundary conditions.
        """
        self.A = self.A[np.ix_(Neumann_dofs, Neumann_dofs)]
        self.f = self.f[Neumann_dofs]

    def linear_system_solver(self, A, f, solver_type=None, precon_type=None, tol=1.0E-06, 
                             maxiter=None):
        """
        Solve the linear system Ax = f using different methods.

        Parameters:
            A: The left hand side matrix.
            f: The right hand side vector.
            solver_type: The type of solver to use (e.g., 'spsolve', 'cg', 'bicg', 'bicgstab', 'gmres').
            precon_type: The type of preconditioner to use (e.g., 'aINV', 'iLU', 'identity').
            tol: The tolerance for linear solver convergence (default is 1.0E-06).
            maxiter: The maximum number of iterations (default is None, which uses the default value for the solver).
        """
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

    @abstractmethod
    def solve(self, *args, **kwargs):
        """
        Solve the system of equations using the specified method.
        """
        pass

class NewtonRaphsonSolver(Solver):

    def __init__(self, system):
        """
        Initialize the Newton-Raphson solver.

        Parameters:
            system: The system containing the weak form, state, and internal forces.
        """
        # invoke the parent (Solver) class
        Solver.__init__(self, system)
        # residual norm at the start of the iterations
        self.initial_residual_norm = 1.0

    def __update_solution(self, solution_increment):
        """
        Update the solution vector with the given increment.

        Parameters:
            solution_increment : The increment to be added to the current solution vector.
        """
        if (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_ADD_ROT):
            self.solution += solution_increment
        elif (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_MUL_ROT):
            # perform additive update for translations
            self.solution[self.translational_dof_indices] += \
                solution_increment[self.translational_dof_indices]
            # perform multiplicative update for rotations
            self.system.weak_form.update_rotational_solution(self.solution, solution_increment)
        else:
            raise NotImplementedError(
                "Solution update type %s is not implemented in the Newton-Raphson solver." % 
                self.system.weak_form.solution_update_type.name)
        # update the internal variables in the weak form
        self.system.weak_form.update_internal_variables(solution_increment)

    def solve(self, Nmax=10, tol=1.0E-05, LSsolver=None, LSprecon=None, LStol=1.0E-06, 
              LSmaxiter=None):
        """
        Solve the system of equations using the Newton-Raphson method.

        Parameters:
            Nmax: The maximum number of iterations (default is 10).
            tol: The tolerance for solver convergence (default is 1.0E-05).
            LSsolver: The linear system solver to use (default is None).
            LSprecon: The preconditioner to use (default is None).
            LStol: The tolerance for the linear system solver (default is 1.0E-06).
            LSmaxiter: The maximum number of iterations for the linear system solver (default is None).
        """
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
        # calculate the initial residual norm
        initial_residual = np.zeros([self.system.nequations, 1])
        self.system.assemble_residual(initial_residual, self.solution, nodal_loads)
        self.initial_residual_norm = np.linalg.norm(initial_residual[Neumann_dofs], ord=2)
        # handle the case when initial residual norm is a very small number
        self.initial_residual_norm = 1.0 if self.initial_residual_norm < 1.0E-20 else self.initial_residual_norm
        # Newton-Raphson iterations
        for i in range(0, Nmax):
            # assemble the linear system
            self.system.assemble(self.A, self.f, self.solution, nodal_loads=nodal_loads)
            # forces to be applied after static condensation of Dirichlet dofs
            if (i == 0): # in the first iteration
                static_condensation_forces = np.matmul(self.A[np.ix_(Neumann_dofs, Dirichlet_dofs)], Dirichlet_solution)
                # add the Dirichlet solution to the overall solution vector
                solution_increment = np.zeros([self.system.nequations, 1])
                solution_increment[Dirichlet_dofs] = Dirichlet_solution
                self.__update_solution(solution_increment)
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
            solution_increment = np.zeros([self.system.nequations, 1])
            solution_increment[Neumann_dofs] = self.linear_system_solver(
                self.A, (self.f)-static_condensation_forces, solver_type=LSsolver, 
                precon_type=LSprecon, tol=LStol, maxiter=LSmaxiter)
            # update the overall solution vector
            self.__update_solution(solution_increment)
            # assess convergence
            # the current residual (= f_ext - f_int)
            updated_residual = np.zeros([self.system.nequations, 1])
            self.system.assemble_residual(updated_residual, self.solution, nodal_loads)
            # the residual norm at all the NEUMANN NODES
            res_L2_norm = np.linalg.norm(updated_residual[Neumann_dofs], ord=2)
            print("\nIteration:", i + 1, ", |R| = %.2e, |R|/|R0| = %.2e" % (res_L2_norm,
                  res_L2_norm/self.initial_residual_norm))
            if ((res_L2_norm <= tol) or ((res_L2_norm/self.initial_residual_norm) <= tol)):
                print("\nSolver converged!!!")
                # update the system attributes
                self.system.update(self.solution)
                return
            else:
                # reset the linear system
                self.reset_system()
        # if the solver did not converge
        sys.exit("\nSolver did not converge after %d iterations." % (Nmax))

class DynamicSolver(Solver):

    def __init__(self, system):
        """
        Initialize the dynamic solver.

        Parameters:
            system: The system containing the weak form, state, and internal forces.
        """
        # invoke the parent (Solver) class
        Solver.__init__(self, system)
        # initialize the mass matrix
        self.M = np.zeros([system.nequations, system.nequations])
        # the "velocity" (linear velocities and time derivative of the tangents)
        self.velocity = np.zeros([system.nequations, 1])
        # the "acceleration" (linear accelerations and double time derivative of the tangents)
        self.acceleration = np.zeros([system.nequations, 1])

    # Function to reset the dynamic linear system
    def reset_system(self):
        """
        Reset the dynamic linear system matrices and vectors.
        """
        super().reset_system()
        self.M.fill(0.0)

    def set_initial_conditions(self, initial_solution, initial_velocity):
        """
        Set the initial conditions for the dynamic solver.

        Parameters:
            initial_solution: The initial solution vector.
            initial_velocity: The initial velocity vector.
        """
        self.solution = copy.deepcopy(np.reshape(initial_solution, [self.system.nequations, 1]))
        self.velocity = copy.deepcopy(np.reshape(initial_velocity, [self.system.nequations, 1]))
        print("\nWARNING: Set boundary conditions before this step to ensure correct "
              "computation of initial accelerations!")
        # compute initial accelerations
        _, Neumann_dofs = self.create_dof_arrays()
        nodal_loads = np.zeros([self.system.nequations, 1])
        nodal_loads[Neumann_dofs] += np.reshape(self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
        # assemble the mass and residual
        self.reset_system()
        mass = np.zeros([self.system.nequations, self.system.nequations])
        self.system.assemble_mass(mass, lump=True)
        self.system.assemble_residual(self.f, self.solution, nodal_loads=nodal_loads)
        # solve for the initial accelerations
        self.acceleration = self.f / np.diag(mass).reshape([-1, 1])
        # update the system attributes
        self.system.update(self.solution)

class ImplicitNewmarkSolver(DynamicSolver):

    def __init__(self, system):
        """
        Initialize the Implicit Newmark solver.

        Parameters:
            system: The system containing the weak form, state, and internal forces.
        """
        # invoke the parent (DynamicSolver) class
        DynamicSolver.__init__(self, system)
        # residual norm at the start of the iterations
        self.initial_residual_norm = 1.0
        # constants for the Newmark time integration scheme
        self.beta = 0.25
        self.gamma = 0.50
        # compute the consistent mass
        self.system.assemble_mass(self.M, lump=False)

    def __initialize_state(self, dt):
        """
        Initialize the solution, velocity and acceleration vectors in the current step.

        Parameters:
            dt: The time step size.
        """
        # constants in the time integration scheme
        c0 = 1.0/(self.beta*(dt**2.0))
        c1 = (1.0-self.gamma)*dt
        c2 = self.gamma*dt
        c3 = (0.5-self.beta)*(dt**2.0)
        if (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_ADD_ROT):
            # initialize the solution, velocity and acceleration vectors
            acceleration_prev = copy.deepcopy(self.acceleration)
            self.acceleration = -((dt*self.velocity)+(c3*acceleration_prev))*c0
            self.velocity += (c1*acceleration_prev) + \
                (c2*self.acceleration)
            ############## For the Dirichlet dofs ##############
            # create the Dirichlet dof array
            Dirichlet_dofs, _ = self.create_dof_arrays()
            # generate the Dirichlet solution vector
            Dirichlet_solution = np.reshape(self.bcvalues, [self.system.nequations, 1])[
                Dirichlet_dofs]
            # update the solution, velocity and acceleration of Dirichlet Dofs
            self.solution[Dirichlet_dofs] += Dirichlet_solution
            self.velocity[Dirichlet_dofs] += Dirichlet_solution*c0*c2
            self.acceleration[Dirichlet_dofs] += Dirichlet_solution*c0
        else:
            raise NotImplementedError(
                "Solution update type %s is not implemented in the Implicit Newmark solver." %
                self.system.weak_form.solution_update_type.name)

    def __update_state(self, dt, solution_increment_neumann):
        """
        Update the solution, velocity and acceleration vectors with the given increment for Neumann dofs.

        Parameters:
            dt : The time step size.
            solution_increment_neumann : The increment to be added to the current solution vector for Neumann dofs.
        """
        # constants in the time integration scheme
        c0 = self.gamma/(self.beta*dt)
        c1 = 1.0/(self.beta*(dt**2.0))
        # create the Neumann dof array
        _, Neumann_dofs = self.create_dof_arrays()
        if (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_ADD_ROT):
            # update the solution, velocity and acceleration of Neumann Dofs
            self.solution[Neumann_dofs] += solution_increment_neumann
            self.velocity[Neumann_dofs] += c0*solution_increment_neumann
            self.acceleration[Neumann_dofs] += c1*solution_increment_neumann
        else:
            raise NotImplementedError(
                "Solution update type %s is not implemented in the Implicit Newmark solver." %
                self.system.weak_form.solution_update_type.name)

    def solve(self, dt, Nmax=10, tol=1.0E-05, LSsolver=None, LSprecon=None, LStol=1.0E-06, 
              LSmaxiter=None):
        """
        Solve the system of equations using the Implicit Newmark time integration scheme.

        Parameters:
            dt: The time step size.
            Nmax: The maximum number of iterations (default is 10).
            tol: The tolerance for solver convergence (default is 1.0E-05).
            LSsolver: The linear system solver to use (default is None).
            LSprecon: The preconditioner to use (default is None).
            LStol: The tolerance for the linear system solver (default is 1.0E-06).
            LSmaxiter: The maximum number of iterations for the linear system solver (default is None).
        """
        # constants in the time integration scheme
        c0 = 1.0/(self.beta*(dt**2.0))
        c1 = 1.0/(self.beta*dt)
        c2 = (1.0/(2.0*self.beta)) - 1.0
        # reset the system before solving
        Solver.reset_system()
        # create the Dirichlet and Neumann global dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # If Dirichlet boundary conditions are not available
        if (Dirichlet_dofs.size == 0):
            sys.exit("\nDirichlet boundary conditions are not found.")
        else:
            # generate the nodal load vector
            nodal_loads = np.zeros([self.system.nequations, 1])
            nodal_loads[Neumann_dofs] += np.reshape(self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
            # initialize the solution, velocity and acceleration vectors
            self.__initialize_state(dt)
        # calculate the initial residual norm
        initial_residual = np.zeros([self.system.nequations, 1])
        self.system.assemble_residual(initial_residual, self.solution, nodal_loads)
        initial_residual -= np.matmul(self.M, self.acceleration)
        self.initial_residual_norm = np.linalg.norm(initial_residual[Neumann_dofs], ord=2)
        # handle the case when initial residual norm is a very small number
        self.initial_residual_norm = 1.0 if self.initial_residual_norm < 1.0E-20 else self.initial_residual_norm
        # Newton-Raphson iterations
        for i in range(0, Nmax):
            # assemble the stiffness matrix and force vector
            self.system.assemble(self.A, self.f, self.solution, nodal_loads = nodal_loads)
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
            solution_increment_neumann = self.linear_system_solver(
                self.A, self.f, solver_type=LSsolver, precon_type=LSprecon, 
                tol=LStol, maxiter=LSmaxiter)
            # update the state of the system
            self.__update_state(dt, solution_increment_neumann)
            # assess convergence
            # the current residual (= f_external - f_internal - f_inertial)
            updated_residual = np.zeros([self.system.nequations, 1])
            self.system.assemble_residual(updated_residual, self.solution, nodal_loads)
            updated_residual -= np.matmul(self.M, self.acceleration)
            # the residual norm at all the NEUMANN NODES
            res_L2_norm = np.linalg.norm(updated_residual[Neumann_dofs], ord=2)
            print("\nIteration:", i + 1, ", |R| = %.2e, |R|/|R0| = %.2e" % (res_L2_norm,
                  res_L2_norm/self.initial_residual_norm))
            if ((res_L2_norm <= tol) or ((res_L2_norm/self.initial_residual_norm) <= tol)):
                print("\nSolver converged!!!")
                # update the system attributes
                self.system.update(self.solution)
                return
            else:
                # reset the linear system
                Solver.reset_system()
        # if the solver did not converge
        sys.exit("\nSolver did not converge after %d iterations." % (Nmax))

class ExplicitNewmarkSolver(DynamicSolver):

    def __init__(self, system):
        """
        Initialize the Explicit Newmark solver.

        Parameters:
            system: The system containing the weak form, state, and internal forces.
        """
        # invoke the parent (Solver) class
        DynamicSolver.__init__(self, system)
        # initialize the stable time step size
        self.stable_time_step = None
        # compute the lumped mass
        self.system.assemble_mass(self.M, lump=True)
        self.lumped_mass = (np.diag(self.M)).reshape([-1, 1])
        # damping factor
        self.__damping_factor = 0.0

    def compute_system_frequencies(self):
        """
        Compute the natural frequencies of the system.

        Returns:
            The natural frequencies of the system.
        """
        # create the Dirichlet and Neumann global dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # if Dirichlet boundary conditions are not available
        if (Dirichlet_dofs.size == 0):
            sys.exit("\nDirichlet boundary conditions are not found.")
        stiffness = np.zeros([self.system.nequations, self.system.nequations])
        residual = np.zeros([self.system.nequations, 1])
        self.system.assemble(stiffness, residual, self.solution, nodal_loads=np.zeros([self.system.nequations, 1]))
        eig_vals, _ = sp.linalg.eig(stiffness[np.ix_(Neumann_dofs, Neumann_dofs)],
                                    self.M[np.ix_(Neumann_dofs, Neumann_dofs)])
        # compute the complex valued Eigen frequencies of the system
        eig_freqs = np.sqrt(eig_vals)
        if (not (((eig_freqs.real > 0.0).all()) and ((eig_freqs.imag >= 0.0).all()))):
            print("\nWARNING: Either the real and/or imaginary parts of the Eigen frequencies are negative.")
        return np.absolute(eig_freqs)

    def set_stable_time_step(self, time_factor=0.90):
        """
        Compute and set the stable time step based on the system frequencies.

        Parameters:
            time_factor: A factor to adjust the stable time step (default is 0.90).
        """
        print("\nRunning stable time computations!!!")
        sys_freqs = self.compute_system_frequencies()
        # get the maximum frequency of the system
        max_sys_freq = sys_freqs[sys_freqs.argmax()]
        self.stable_time_step = time_factor*(2.0/(max_sys_freq.real))

    def activate_damping(self, damping_factor):
        """
        Activate damping in the solver.

        Parameters:
            damping_factor: The damping factor.
        """
        self.__damping_factor = damping_factor

    def set_boundary_conditions(self, bctypes, bcvalues):
        """
        Set the boundary conditions and the stable time step.

        Parameters:
            bctypes: The boundary condition types (0 = Neumann, 1 = Dirichlet).
            bcvalues: The boundary condition values.
        """
        super().set_boundary_conditions(bctypes, bcvalues)
        self.set_stable_time_step()

    def __initialize_state(self, dt):
        """
        Initialize the solution, velocity and acceleration vectors in the current step.

        Parameters:
            dt : The time step size.
        """
        # create the Dirichlet and Neumann dof arrays
        Dirichlet_dofs, Neumann_dofs = self.create_dof_arrays()
        # compute the solution increment in the current step
        solution_increment = np.zeros_like(self.solution)
        solution_increment[Dirichlet_dofs] += np.reshape(
            self.bcvalues, [self.system.nequations, 1])[Dirichlet_dofs]
        # NOTE: The update equations for rotational dofs (also in the case of multiplicative 
        # updates) can be the same as the translational dofs as per Marino et al., CMAME, 2019 
        # and Krysl et al., IJNME. 2005.
        solution_increment[Neumann_dofs] += (dt*self.velocity[Neumann_dofs]) + (
            ((dt**2.0)/2.0)*self.acceleration[Neumann_dofs])
        ######### Update the solution #########
        if (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_ADD_ROT):
            # perform additive update for translations and rotations
            self.solution += solution_increment
        elif (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_MUL_ROT):
            # perform additive update for translations
            self.solution[self.translational_dof_indices] += \
                solution_increment[self.translational_dof_indices]
            # perform multiplicative update for rotations
            self.system.weak_form.update_rotational_solution(
                self.solution, solution_increment)
        else:
            raise NotImplementedError(
                "Solution update type %s is not implemented in the Explicit Newmark solver." %
                self.system.weak_form.solution_update_type.name)
        # update the internal variables in the weak form
        self.system.weak_form.update_internal_variables(solution_increment)
        ######### Update the velocity and acceleration (only of Dirichlet dofs) #########
        # NOTE: The following equations might be incorrect for angular velocities of
        # Dirichlet dofs in the case of multiplicative rotation updates. This is still 
        # fine if we have homogeneous Dirichlet boundary conditions. In the case of 
        # inhomogeneous Dirichlet boundary conditions, these relations need to be modified!!!
        # for Dirichlet dofs
        velocity_prev_Dirichlet = copy.deepcopy(
            self.velocity[Dirichlet_dofs])
        self.velocity[Dirichlet_dofs] = solution_increment[Dirichlet_dofs]/dt
        self.acceleration[Dirichlet_dofs] = (
            self.velocity[Dirichlet_dofs] - velocity_prev_Dirichlet)/dt
        # for Neumann dofs
        self.velocity[Neumann_dofs] += ((dt/2.0)
                                        * self.acceleration[Neumann_dofs])

    def __update_state(self, dt, accelerations_neumann):
        """
        Update the velocity and acceleration vectors of the Neumann dofs.

        Parameters:
            dt : The time step size.
            accelerations_neumann : The accelerations to be applied to the Neumann dofs.
        """
        # create the Neumann dof array
        _, Neumann_dofs = self.create_dof_arrays()
        if ((self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_ADD_ROT) or
            (self.system.weak_form.solution_update_type == SolutionUpdateType.ADD_TNS_MUL_ROT)):
            # update the velocity and acceleration of Neumann Dofs
            self.acceleration[Neumann_dofs] = accelerations_neumann
            self.velocity[Neumann_dofs] += ((dt/2.0)
                                            * self.acceleration[Neumann_dofs])
        else:
            raise NotImplementedError(
                "Solution update type %s is not implemented in the Explicit Newmark solver." %
                self.system.weak_form.solution_update_type.name)

    def solve(self, dt):
        """
        Solve the system of equations using the Explicit Newmark time integration scheme.

        Parameters:
            dt: The time step size.
        """
        # if the time step size input is not provided
        if (dt == None):
            dt = self.stable_time_step
        # perform stability check
        elif (dt > self.stable_time_step):
            sys.exit("\nThe chosen time step size makes the solver unstable in time.")
        # reset the system before solving
        Solver.reset_system(self)
        # create the Dirichlet and Neumann global dof arrays
        _, Neumann_dofs = self.create_dof_arrays()
        # generate the nodal load vector
        nodal_loads = np.zeros([self.system.nequations, 1])
        nodal_loads[Neumann_dofs] += np.reshape(
            self.bcvalues, [self.system.nequations, 1])[Neumann_dofs]
        # the PREDICTOR
        self.__initialize_state(dt)
        # assemble the residual
        self.system.assemble_residual(
            self.f, self.solution, nodal_loads, element_loads=None, update_internal=True)
        # reassemble the mass matrix if weak form is ShearFlexibleGeometricallyExactWeakFormCG
        # or ShearFlexibleGeometricallyExactWeakFormDG
        if ((type(self.system.weak_form) == ShearFlexibleGeometricallyExactWeakFormCG) or
            (type(self.system.weak_form) == ShearFlexibleGeometricallyExactWeakFormDG)):
            self.M.fill(0.0)
            self.system.assemble_mass(
                self.M, lump=True, dt=dt, system_velocities=self.velocity, residual_vector=self.f)
            self.lumped_mass = (np.diag(self.M)).reshape([-1, 1])
        # add mass proportional damping forces to the residual
        self.f -= (self.__damping_factor *
                   (self.lumped_mass / dt)) * self.velocity
        # the CORRECTOR
        # solve the semi-discrete SOE for accelerations of the Neumann Dofs
        accelerations_neumann = self.f[Neumann_dofs]/self.lumped_mass[Neumann_dofs]
        self.__update_state(dt, accelerations_neumann)
        # update the system attributes
        self.system.update(self.solution)
