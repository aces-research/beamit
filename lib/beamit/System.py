import sys
import numpy as np
from beamit import FunctionSpace
from beamit import WeakForm

def cross_op(arr1: np.ndarray, arr2: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    return np.cross(arr1, arr2, axisa=a, axisb=b, axisc=c)

class System:
    
    def __init__(self, function_space, material, betaP = 10.0, betaT = 10.0):
        # use TFKL Geometrically Exact Function Space for TFKL geometrically exact function space
        if (type(function_space) == FunctionSpace.TFKLGeometricallyExactFunctionSpace):
            if (function_space.discretization_type == "CG"):
                # the continuous Galerkin weak form
                self.weak_form = WeakForm.TFKLGeometricallyExactWeakFormCG(function_space, material)
            elif (function_space.discretization_type == "DG"):
                # the discontinuous Galerkin weak form
                self.weak_form = WeakForm.TFKLGeometricallyExactWeakFormDG(
                    function_space, material, betaP, betaT)
            else:
                sys.exit("\nBeam KLTF weak form of the discretization is not available.")
            # the initial state of the system
            self.state = self.initialize_state()
        # use Euler-Bernoulli (EB) weak form for EB function space
        elif (type(function_space) == FunctionSpace.EulerBernoulliFunctionSpace):
            if (function_space.discretization_type == "CG"):
                # the continuous Galerkin weak form
                self.weak_form = WeakForm.EulerBernoulliWeakFormCG(function_space, material)
            elif (function_space.discretization_type == "DG"):
                self.weak_form = WeakForm.EulerBernoulliWeakFormDG(function_space, material, betaP)
            else:
                sys.exit("\nEuler-Bernoulli weak form of the discretization is not available.")
            self.state = np.zeros([self.weak_form.function_space.N, self.weak_form.function_space.dof])
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

    def assemble_stiffness(self, A, solution, nodal_loads, element_loads=None):
        self.weak_form.compute_system_stiffness(A, solution, nodal_loads, element_loads)
        if ((type(self.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormCG) or
                (type(self.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormDG)):
            dofs = self.weak_form.function_space.dof
            # contribution of external nodal moments to the system stiffness
            for i in range(0, self.weak_form.function_space.N):
                nodal_tangents = solution[(dofs*i)+3:(dofs*i)+6, :]
                nodal_tangents_L2 = np.linalg.norm(nodal_tangents, ord=2, axis=0, keepdims=True)
                nodal_moments = nodal_loads[(dofs*i)+3:(dofs*i)+6, :]
                nodal_rp_dyd_rp = np.matmul(nodal_tangents, np.transpose(nodal_tangents))
                nodal_dt4dd_coeff = (np.eye(self.weak_form.function_space.dim)/(nodal_tangents_L2**2.0)) - ((2.0*nodal_rp_dyd_rp)/(nodal_tangents_L2**4.0))
                A[(dofs*i)+3:(dofs*i)+6, (dofs*i)+3:(dofs*i)+6] -= cross_op(nodal_moments, nodal_dt4dd_coeff, 0, 0, 0)

    def assemble_residual(self, f, solution, nodal_loads, element_loads=None, update_internal=False):
        self.weak_form.compute_system_residual(
            f, solution, nodal_loads, element_loads, update_internal)
        # add nodal forces to the residual
        if ((type(self.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormCG) or
                (type(self.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormDG)):
            dofs = self.weak_form.function_space.dof
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
        else:
            f += nodal_loads

    def assemble_mass(self, M, solution):
        self.weak_form.compute_system_mass(M, solution)

    def assemble(self, A, f, solution, nodal_loads, element_loads=None):
        # assemble stiffness
        self.assemble_stiffness(A, solution, nodal_loads, element_loads)

        # assemble residual
        self.assemble_residual(f, solution, nodal_loads, element_loads)

    # Function to update the variables in the system
    def update(self, solution):
        # update the state of the system
        self.state = np.reshape(solution, [self.weak_form.function_space.N, \
                                self.weak_form.function_space.dof])
        # update the internal forces
        internal_force_vector = np.zeros([self.nequations, 1])
        self.weak_form.compute_system_nodal_forces(
            internal_force_vector, solution, element_loads=None)
        self.internal_forces = np.reshape(internal_force_vector, [
                                          self.weak_form.function_space.N, self.weak_form.function_space.dof])
