import sys
import numpy as np
from beamit import WeakForm

class System:
    
    def __init__(self, function_space, material):
        if (function_space.discretization_type == "CG"):
            # the continuous Galerkin weak form
            self.weak_form = WeakForm.WeakFormCG(function_space, material)
        else:
            sys.exit("\nWeak form of the discretization is not available.")
        # the initial state of the system
        self.state = self.initialize_state()
        # the number of equations
        self.nequations = self.weak_form.function_space.N*self.weak_form.function_space.dof

    # Function to initialize the state of the system
    def initialize_state(self):
        initial_state = np.zeros([self.weak_form.function_space.N, self.weak_form.function_space.dof])
        # set the initial positions as the nodal co-ordinates
        initial_state[:, 0:3] = self.weak_form.function_space.nodes
        # assuming initially straight beams are along the x-axis!!!
        initial_state[:, 3:4] = 1.0
        return initial_state

    def assemble_stiffness(self, A, solution):
        self.weak_form.compute_system_stiffness(A, solution)
        pass

    def assemble_residual(self, f, solution, nodal_loads, element_loads_info):
        dofs = self.weak_form.function_space.dof
        self.weak_form.compute_system_residual(f, solution, element_loads_info)
        updated_nodal_loads = np.zeros(nodal_loads.shape)
        for i in range(0, self.weak_form.function_space.N):
            updated_nodal_loads[dofs*i:(dofs*i)+3, :] += nodal_loads[dofs*i:(dofs*i)+3, :]
            nodal_tangents = solution[(dofs*i)+3:(dofs*i)+6, :]
            nodal_tangents_L2 = np.linalg.norm(nodal_tangents, ord=2, axis=0, keepdims=True)
            t4_nodal = nodal_tangents/(nodal_tangents_L2**2.0)
            # compute the cross-product for the nodal moments
            updated_nodal_loads[(dofs*i)+3:(dofs*i)+6, :] += \
                    np.cross(nodal_loads[(dofs*i)+3:(dofs*i)+6, :], t4_nodal, axisa=0, axisb=0, axisc=0)
        f += updated_nodal_loads
        pass

    def assemble(self, A, f, solution, nodal_loads = 0.0, element_loads_info = None):
        # assemble stiffness
        self.assemble_stiffness(A, solution)

        # assemble residual
        self.assemble_residual(f, solution, nodal_loads, element_loads_info)

    # Function to update the state of the system
    def update(self, solution):
        self.state = np.reshape(solution, [self.weak_form.function_space.N, \
                                self.weak_form.function_space.dof])
        pass