import sys
import numpy as np
from lib.sgi import WeakForm

class System:
    
    def __init__(self, function_space, material):
        if (function_space.discretization_type == "CG"):
            # the continuous Galerkin weak form
            self.weak_form = WeakForm.WeakFormCG(function_space, material)
        else:
            sys.exit("\nWeak form of the discretization is not available.")
        # the initial state of the system
        self.nodal_coordinates = self.weak_form.function_space.nodes
        # the number of equations
        self.nequations = self.weak_form.function_space.N*self.weak_form.function_space.dof

    def assemble_stiffness(self, A, solution):
        self.weak_form.compute_system_stiffness(A, solution)
        pass

    def assemble_residual(self, f, solution, nodal_loads, element_loads_info):
        self.weak_form.compute_system_residual(f, solution, element_loads_info)
        f += nodal_loads
        pass

    def assemble(self, A, f, solution, nodal_loads = 0.0, element_loads_info = None):
        # assemble stiffness
        self.assemble_stiffness(A, solution)

        # assemble residual
        self.assemble_residual(f, solution, nodal_loads, element_loads_info)
    
    # Function to update the node positions (nodal co-ordinates)
    def update_node_positions(self, solution):
        solution_matrix = np.reshape(solution, [self.weak_form.function_space.N, \
                                self.weak_form.function_space.dof])
        # first three columns in the solution matrix are the new positions
        self.nodal_coordinates = solution_matrix[:, 0:3]
        pass

    # Function to update the attributes of the system
    def update(self, solution):
        # update the node positions
        self.update_node_positions(solution)
        pass