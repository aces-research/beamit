import sys
import numpy as np
from beamit import FunctionSpace
from beamit import WeakForm

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

    def assemble_residual(self, f, solution, nodal_loads, element_loads=None, update_internal=False):
        self.weak_form.compute_system_residual(
            f, solution, nodal_loads, element_loads, update_internal)
        # add nodal loads to the residual
        # Generally, the addition of nodal loads just involves a direction addition to the residual. 
        # But for the TFKL geometrically exact weak form, a special treatment is needed which is 
        # implemented in the following function in the respective weak form
        self.weak_form.add_nodal_loads_to_residual(f, solution, nodal_loads)

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
