# !/usr/bin/env python3
#
# Copyright (c) 2025, the beamit authors, all rights reserved.
#

import sys
import numpy as np
from beamit.FunctionSpace import *
from beamit.WeakForm.TFKLGeometricallyExactWeakForm import *
from beamit.WeakForm.EulerBernoulliWeakForm import *
from beamit.WeakForm.ShearFlexibleGeometricallyExactWeakForm import *

class System:

    def __init__(self, function_space, material, betaP = 10.0, betaT = 10.0):
        """
        Initialize the system with the given function space, material, and DG penalty parameters.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties.
            betaP: DG penalty parameter for translational compatibility (default is 10.0).
            betaT: DG penalty parameter for rotational compatibility (default is 10.0).
        """
        # use TFKL geometrically exact weak form for TFKL geometrically exact function space
        if (type(function_space) == TFKLGeometricallyExactFunctionSpace):
            if (function_space.discretization_type == "CG"):
                # the continuous Galerkin weak form
                self.weak_form = TFKLGeometricallyExactWeakFormCG(function_space, material)
            elif (function_space.discretization_type == "DG"):
                # the discontinuous Galerkin weak form
                self.weak_form = TFKLGeometricallyExactWeakFormDG(
                    function_space, material, betaP, betaT)
            else:
                sys.exit(
                    "\nTFKL geometrically exact weak form of the discretization is not available.")
            # the initial state of the system
            self.state = np.zeros(
                [self.weak_form.function_space.N, self.weak_form.function_space.dof])
            self.state[:, 0:3] = self.weak_form.function_space.nodes
            # assuming initially straight beams are along the x-axis!!!
            self.state[:, 3:4] = 1.0
        # use Euler-Bernoulli (EB) weak form for EB function space
        elif (type(function_space) == EulerBernoulliFunctionSpace):
            if (function_space.discretization_type == "CG"):
                # the continuous Galerkin weak form
                self.weak_form = EulerBernoulliWeakFormCG(function_space, material)
            elif (function_space.discretization_type == "DG"):
                self.weak_form = EulerBernoulliWeakFormDG(function_space, material, betaP)
            else:
                sys.exit("\nEuler-Bernoulli weak form of the discretization is not available.")
            # the initial state of the system
            self.state = np.zeros(
                [self.weak_form.function_space.N, self.weak_form.function_space.dof])
        # use shear flexible geometrically exact weak form for shear flexible geometrically exact function space
        elif (type(function_space) == ShearFlexibleGeometricallyExactFunctionSpace):
            if (function_space.discretization_type == "CG"):
                # the continuous Galerkin weak form
                self.weak_form = ShearFlexibleGeometricallyExactWeakFormCG(function_space, material)
            elif (function_space.discretization_type == "DG"):
                # the discontinuous Galerkin weak form
                self.weak_form = ShearFlexibleGeometricallyExactWeakFormDG(
                    function_space, material, betaP, betaT)
            else:
                sys.exit(
                    "\nShear flexible geometrically exact weak form of the discretization is not available.")
            # the initial state of the system
            self.state = np.zeros(
                [self.weak_form.function_space.N, self.weak_form.function_space.dof])
            self.state[:, 0:3] = self.weak_form.function_space.nodes
        # the internal forces of the system
        self.internal_forces = np.zeros(
            [self.weak_form.function_space.E*self.weak_form.function_space.npel, 
             self.weak_form.function_space.dof])
        # the number of equations
        self.nequations = self.weak_form.function_space.N*self.weak_form.function_space.dof

    def assemble_stiffness(self, A, solution, nodal_loads, element_loads=None):
        """
        Assemble the stiffness matrix for the system.

        Parameters:
            A: The stiffness matrix to be assembled.
            solution: The current solution vector.
            nodal_loads: Nodal loads to be applied.
            element_loads: Element loads to be applied (optional).
        """
        self.weak_form.compute_system_stiffness(A, solution, nodal_loads, element_loads)

    def assemble_residual(self, f, solution, nodal_loads, element_loads=None, 
                          update_internal=False):
        """
        Assemble the residual vector for the system.

        Parameters:
            f: The residual vector to be assembled.
            solution: The current solution vector.
            nodal_loads: Nodal loads to be applied.
            element_loads: Element loads to be applied (optional).
            update_internal: Whether to update the internal variables (default is False).
        """
        self.weak_form.compute_system_residual(
            f, solution, element_loads, update_internal)
        # add nodal loads to the residual
        # NOTE: Generally, the addition of nodal loads just involves a direction addition to the 
        # residual. But for some weak forms, a special treatment is needed which is implemented in
        # the following function in the respective weak form
        self.weak_form.add_nodal_loads_to_residual(f, solution, nodal_loads)

    def assemble_mass(self, M, **kwargs):
        """
        Assemble the mass matrix for the system.

        Parameters:
            M: The mass matrix to be assembled.
            **kwargs: Optional keyword arguments.
        """
        self.weak_form.compute_system_mass(M, **kwargs)

    def assemble(self, A, f, solution, nodal_loads, element_loads=None):
        """
        Assemble the stiffness matrix and residual vector for the system.

        Parameters:
            A: The stiffness matrix to be assembled.
            f: The residual vector to be assembled.
            solution: The current solution vector.
            nodal_loads: Nodal loads to be applied.
            element_loads: Element loads to be applied (optional).
        """
        # assemble stiffness
        self.assemble_stiffness(A, solution, nodal_loads, element_loads)

        # assemble residual
        self.assemble_residual(f, solution, nodal_loads, element_loads)

    def update(self, solution):
        """
        Update the state variables of the system with the new solution.

        Parameters:
            solution: The new solution vector.
        """
        # update the state of the system
        self.state = np.reshape(solution, self.state.shape)
        # update the internal forces
        internal_force_vector = np.zeros([self.internal_forces.size, 1])
        self.weak_form.compute_system_nodal_forces(
            internal_force_vector, solution, element_loads=None)
        self.internal_forces = np.reshape(internal_force_vector, self.internal_forces.shape)
