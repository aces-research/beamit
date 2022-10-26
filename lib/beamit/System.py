import sys
import numpy as np
from beamit import WeakForm

def cross_op(arr1:np.ndarray,arr2:np.ndarray,a:int,b:int,c:int)->np.ndarray:
    return np.cross(arr1,arr2,axisa=a,axisb=b,axisc=c)

class System:
    
    def __init__(self, function_space, material, betaP = 10.0, betaT = 10.0):
        if (function_space.discretization_type == "CG"):
            # the continuous Galerkin weak form
            self.weak_form = WeakForm.WeakFormCG(function_space, material)
        elif (function_space.discretization_type == "DG"):
            # the discontinuous Galerkin weak form
            self.weak_form = WeakForm.WeakFormDG(function_space, material, betaP, betaT)
        else:
            sys.exit("\nWeak form of the discretization is not available.")
        # the initial state of the system
        self.state = self.initialize_state()
        # the internal forces of the system
        self.internal_forces = np.zeros([self.weak_form.function_space.N, self.weak_form.function_space.dof])
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
                        cross_op(nodal_loads[(dofs*i)+3:(dofs*i)+6, :], t4_nodal, 0, 0, 0)
        f += updated_nodal_loads
        pass

    def assemble_mass(self, M):
        self.weak_form.compute_system_mass(M)
        pass

    def assemble(self, A, f, solution, nodal_loads = 0.0, element_loads_info = None):
        # assemble stiffness
        self.assemble_stiffness(A, solution)

        # assemble residual
        self.assemble_residual(f, solution, nodal_loads, element_loads_info)

    # Function to update the variables in the system
    def update(self, solution):
        # update the state of the system
        self.state = np.reshape(solution, [self.weak_form.function_space.N, \
                                self.weak_form.function_space.dof])
        internal_force_vector = np.zeros([self.nequations, 1])
        self.weak_form.compute_system_internal_forces(internal_force_vector, solution)
        # compute the cross-product for the moments
        updated_internal_force_vector = np.zeros([self.nequations, 1])
        dofs = self.weak_form.function_space.dof
        for i in range(0, self.weak_form.function_space.N):
            updated_internal_force_vector[dofs*i:(dofs*i)+3, :] += internal_force_vector[dofs*i:(dofs*i)+3, :]
            nodal_tangents = solution[(dofs*i)+3:(dofs*i)+6, :]
            nodal_tangents_L2 = np.linalg.norm(nodal_tangents, ord=2, axis=0, keepdims=True)
            t4_nodal = nodal_tangents/(nodal_tangents_L2**2.0)
            updated_internal_force_vector[(dofs*i)+3:(dofs*i)+6, :] += \
                cross_op(internal_force_vector[(dofs*i)+3:(dofs*i)+6, :], t4_nodal, 0, 0, 0)
        self.internal_forces = np.reshape(updated_internal_force_vector, [self.weak_form.function_space.N, \
                                self.weak_form.function_space.dof])
        pass