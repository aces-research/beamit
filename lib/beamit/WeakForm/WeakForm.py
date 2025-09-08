from abc import ABC, abstractmethod
import numpy as np
from beamit.WeakForm.Utils import SolutionUpdateType


class WeakForm(ABC):

    def __init__(self, function_space, material):
        """
        Initialize the WeakForm (abstract) class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
        """
        # the geometrical information
        self.function_space = function_space
        # the physical information
        self.material = material
        # the type of update to be applied to the solution
        self.solution_update_type = SolutionUpdateType.ADD_TNS_ADD_ROT

    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        """
        Compute the system residual based on the provided unknowns and element loads information.

        Parameters:
            f: The residual vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
            update_internal: If True, update the internal variables in the weak form.
        """
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten(
            )
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads == None):  # No element loads
                f[global_element_dofs] -= self.compute_element_internal_forces(
                    element_unknowns)

    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        """
        Compute the system stiffness matrix based on the provided unknowns and element loads information.

        Parameters:
            A: The stiffness matrix to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
            element_loads: The distributed loads on the elements.
        """
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten(
            )
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads == None):  # No element loads
                A[np.ix_(global_element_dofs, global_element_dofs)
                  ] += self.compute_element_internal_stiffness(element_unknowns)

    @abstractmethod
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads):
        """
        Compute the system nodal forces based on the provided unknowns and element loads.

        Parameters:
            f: The force vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
        """
        pass

    @abstractmethod
    def compute_system_mass(self, M, lump=True, **kwargs):
        """
        Compute the system mass matrix based on the provided solution vector.

        Parameters:
            M: The mass matrix to be assembled.
            lump: Whether to use lumped mass matrix (default is True).
            **kwargs: Optional keyword arguments.
        """
        pass

    def add_nodal_loads_to_residual(self, f, system_unknowns, nodal_loads):
        """
        Add the nodal loads to the residual vector.

        Parameters:
            f: The residual vector to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
        """
        f += nodal_loads

    @abstractmethod
    def compute_element_internal_forces(self, element_unknowns):
        """
        Compute the internal forces for an element based on its unknowns.

        Parameters:
            system_unknowns: The unknowns of the system.
        """
        pass

    @abstractmethod
    def compute_element_internal_stiffness(self, element_unknowns):
        """
        Compute the internal stiffness for an element based on its unknowns.

        Parameters:
            system_unknowns: The unknowns of the system.
        """
        pass

    def update_rotational_solution(self, system_unknowns, solution_increment):
        """
        Update the rotational degrees of freedom in the solution based on the solution increment.

        Parameters:
            system_unknowns: The unknowns of the system.
            solution_increment: The increment in the solution.
        """
        raise NotImplementedError(
            "The update_rotational_solution method is called from the base class. " \
            " Implement it in the derived class.")

    def update_internal_variables(self, system_unknowns_increment):
        """
        Update the internal variables in the weak form based on the increment in the system unknowns.

        Parameters:
            system_unknowns_increment: The increment in the system unknowns.
        """
        pass
