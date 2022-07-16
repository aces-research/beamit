import numpy as np
import sys

class NewtonRaphsonSolver:

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
        # boundary condition types (0 = Dirichlet, 1 = Neumann) matrix
        self.bctypes = np.ones([self.system.weak_form.function_space.N, self.system.weak_form.function_space.dof], dtype=np.int64)
        # boundary condition values matrix
        self.bcvalues = np.zeros([self.system.weak_form.function_space.N, self.system.weak_form.function_space.dof])

    # Function to initialize unknowns in the system to the undeformed state of the beam
    def initialize(self):
        # assuming initially straight beams are along the x-axis!!!
        nodes_x = self.system.weak_form.function_space.nodes[:, 0:1]
        dofs = self.system.weak_form.function_space.dof
        for i in range(0, nodes_x.shape[0]):
            self.solution[dofs*i:(dofs*i)+1, :] = nodes_x[i:i+1, :]
        pass
    
    # Function to reset the linear system
    def reset_system(self):
        self.A = np.zeros([self.system.nequations, self.system.nequations])
        self.f = np.zeros([self.system.nequations, 1])
        pass

    # Create arrays with entries being the global dof numbers of Dirichlet and Neumann dofs
    def create_dof_arrays(self):
        bctypes_vec = np.reshape(self.bctypes, [self.system.nequations, 1])
        global_dofs = np.arange(0, self.system.nequations, 1)
        Dirichlet_dofs = global_dofs[(bctypes_vec == 0).flatten()]
        Neumann_dofs = global_dofs[(bctypes_vec == 1).flatten()]
        return Dirichlet_dofs, Neumann_dofs
    
    # Function to apply the Dirichlet boundary conditions by static condensation
    def apply_static_condensation(self, Neumann_dofs):
        self.A = self.A[np.ix_(Neumann_dofs, Neumann_dofs)]
        self.f = self.f[Neumann_dofs]
    
    def solve(self, Nmax = 10, tol = 1.0E-05):
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
            # add the Dirichlet solution to the overall solution vector
            self.solution[Dirichlet_dofs] += Dirichlet_solution
        # Newton-Raphson iterations
        for i in range(0, Nmax):
            # assemble the linear system
            self.system.assemble(self.A, self.f, self.solution, nodal_loads = nodal_loads)
            # apply the Dirichlet BCs
            self.apply_static_condensation(Neumann_dofs)
            # checks in the first iteration
            if (i == 0):
                # not enough fixity in the system
                if (np.linalg.det(self.A) == 0.0):
                    sys.exit("\nSystem is not fixed properly.")
                # if loads are applied on the system
                # include a check for elemental loads later!!!
                elif (np.linalg.norm(nodal_loads, ord=2) == 0.0):
                    sys.exit("\nSystem is not loaded.")
                else:
                    print("\nStarting the Newton-Raphson iterations!!!")
            # checks after the first iteration
            else:
                # instability in the system
                if (np.linalg.det(self.A) == 0.0):
                    sys.exit("\nSystem is unstable.")
            # solve the linear system
            solution_increment = np.linalg.solve(self.A, self.f)
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
        # update the system attributes
        self.system.update(self.solution)