import numpy as np
import quaternion
from beamit.WeakForm.WeakForm import WeakForm
from beamit.WeakForm.Utils import SolutionUpdateType, skew_symmetric_matrices


class ShearFlexibleGeometricallyExactWeakFormCG(WeakForm):

    def __init__(self, function_space, material):
        """
        Initialize the ShearFlexibleGeometricallyExactWeakFormCG class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
        """
        # initialize the parent (WeakForm) class
        WeakForm.__init__(self, function_space, material)
        # the type of update to be applied for the solution
        self.solution_update_type = SolutionUpdateType.ADD_TNS_MUL_ROT
        # containers to store the orientation and curvature of the beam at each quadrature point
        # NOTE: Since we consider here that the beam is initially straight and aligned with the 
        # x-axis, the orientation and curvature are initialized to zero.
        self.orientation = np.zeros(
            [self.function_space.E, self.function_space.Q, 
             self.function_space.dim])  # saved as rotation vectors
        self.curvature = np.zeros(
            [self.function_space.E, self.function_space.Q, self.function_space.dim])
        # container to store the curvature at the nodes
        # NOTE: The curvature at the nodes is not used in the weak form, but it is needed for
        # post-processing (to compute the internal moments).
        self.curvature_nodes = np.zeros(
            [self.function_space.N, self.function_space.dim])

    @staticmethod
    def _compute_transformation_matrix(orientation):
        """
        Compute the transformation matrix (T) which converts additive updates to multiplicative updates.

        The transformation matrix is computed based on the current orientation of the beam.

        Parameters:
            orientation: The orientation in terms of rotation vectors. **orienation** is assumed to be of shape (n, 3) where n is the number of rotation vectors.
        Returns:
            T: The transformation matrix of size (n, 3, 3).
        """
        # check if the orientation is of size (n, 3)
        if (orientation.ndim != 2 or orientation.shape[1] != 3):
            raise ValueError(
                "Orientation must be of shape (n, 3), where n is the number of rotation vectors.")
        psi_norm = np.linalg.norm(orientation, axis=1)
        I = np.eye(3)[None, :, :]
        # avoid division by zero
        small_norm_idxs = psi_norm < 1.0e-10
        sin_psi = np.sin(psi_norm)
        cos_psi = np.cos(psi_norm)
        psi_norm_safe = np.where(small_norm_idxs, 1.0, psi_norm)
        sin_term = sin_psi / psi_norm_safe
        cos_term = (1.0 - cos_psi) / psi_norm_safe**2
        extra_term = (psi_norm - sin_psi) / psi_norm_safe**3
        # apply limit values for small angles
        sin_term[small_norm_idxs] = 1.0
        cos_term[small_norm_idxs] = 0.50
        extra_term[small_norm_idxs] = 1.0 / 6.0
        sin_term = sin_term[:, None, None]
        cos_term = cos_term[:, None, None]
        extra_term = extra_term[:, None, None]
        # skew symmetric matrices
        S = skew_symmetric_matrices(orientation)
        # outer product
        outer = orientation[:, :, None] * orientation[:, None, :]
        return sin_term * I + cos_term * S + extra_term * outer

    def _compute_element_dof_derivatives(self, e, element_dof_values, local_dof_indices, 
                                         location="Quads"):
        """
        Compute the derivatives of the element dofs with respect to the arclength parameter.

        Parameters:
            e: The element index.
            element_dof_values: The values of the dofs of the element.
            local_dof_indices: The local (translational or rotational) dof indices of the element.
            location: The location where the derivatives should be computed.
                If "Quads", the derivatives are computed at the quadrature points of the element.
                If "Nodes", the derivatives are computed at the nodes of the element.
        Returns:
            element_dof_derivatives: The derivatives of the element dofs with respect to the arclength parameter.
        """
        if (location != "Quads" and location != "Nodes"):
            raise ValueError("Location must be either 'Quads' or 'Nodes'.")
        if (location == "Quads"):
            Np = self.function_space.shape_first_gradients * \
                (1.0/self.function_space.jacobian)
        elif (location == "Nodes"):
            # NOTE: Here we are assuming that there are only two nodes per element!!!
            _, Nxi_left_node = self.function_space.compute_shapes(-1.0)
            Np_left_node = Nxi_left_node*(1.0/self.function_space.jacobian)
            _, Nxi_right_node = self.function_space.compute_shapes(1.0)
            Np_right_node = Nxi_right_node*(1.0/self.function_space.jacobian)
            Np = np.stack([Np_left_node, Np_right_node], axis=0)
        return np.matmul(Np, element_dof_values[local_dof_indices])

    def update_bulk_internal_variables(self, system_unknowns_increment):
        """
        Update the internal variables in the weak form based on the increment in the system unknowns.
        
        The orientation and curvature of the beam at each quadrature point are updated based on 
        the current system unknowns increment.

        Parameters:
            system_unknowns_increment: The increment in the system unknowns.
        """
        N = self.function_space.shape_functions
        # update the orientation and curvature at quadrature points
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_rotation_increment = system_unknowns_increment[
                global_element_dofs][self.function_space.local_rotational_dofs]
            # compute the rotation increment and its derivative
            dtheta = np.matmul(N, element_rotation_increment)[..., 0]
            dtheta_prime = self._compute_element_dof_derivatives(
                i, system_unknowns_increment[global_element_dofs], 
                self.function_space.local_rotational_dofs)[..., 0]
            ############# update of the orientation #############
            current_orientation = self.orientation[i, :, :]
            # NOTE: Careful with the order of multiplication here since quaternion multiplication 
            # is not commutative!
            updated_orientation_quats = \
                quaternion.from_rotation_vector(dtheta) * quaternion.from_rotation_vector(
                    current_orientation)
            # convert the updated orientation quaternions to rotation vectors
            updated_orientation = quaternion.as_rotation_vector(updated_orientation_quats)
            self.orientation[i, :, :] = updated_orientation
            ############# update the curvature #############
            # compute the transformation matrix
            transformation_matrix = self._compute_transformation_matrix(dtheta)
            incremental_rotation_tensor = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta))
            current_curvature = self.curvature[i, :, :]
            self.curvature[i, :, :] = np.matmul(
                transformation_matrix, dtheta_prime[..., None])[..., 0] + \
                np.matmul(incremental_rotation_tensor,
                          current_curvature[..., None])[..., 0]
        # update the curvature at the nodes
        # NOTE: Here we assume that there are only two nodes per element!!!
        N_left_node, _ = self.function_space.compute_shapes(-1.0)
        N_right_node, _ = self.function_space.compute_shapes(1.0)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_rotation_increment = system_unknowns_increment[
                global_element_dofs][self.function_space.local_rotational_dofs]
            # compute the rotation increment and its derivative
            dtheta_left_node = np.matmul(
                N_left_node, element_rotation_increment)[..., 0]
            dtheta_right_node = np.matmul(
                N_right_node, element_rotation_increment)[..., 0]
            dtheta_prime_node = self._compute_element_dof_derivatives(
                i, system_unknowns_increment[global_element_dofs],
                self.function_space.local_rotational_dofs, location="Nodes")[..., 0]
            dtheta_prime_left_node = dtheta_prime_node[0, :]
            dtheta_prime_right_node = dtheta_prime_node[1, :]
            # transformation matrices
            transformation_matrix_left_node = self._compute_transformation_matrix(
                dtheta_left_node[None, ...])[0, ...]
            transformation_matrix_right_node = self._compute_transformation_matrix(
                dtheta_right_node[None, ...])[0, ...]
            incremental_rotation_tensor_left_node = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta_left_node))
            incremental_rotation_tensor_right_node = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta_right_node))
            if (i == 0):  # only for the first element
                # update the curvature at the left node
                self.curvature_nodes[0, :] = np.matmul(
                    transformation_matrix_left_node, dtheta_prime_left_node[..., None])[..., 0] + \
                    np.matmul(incremental_rotation_tensor_left_node,
                              self.curvature_nodes[0, :][..., None])[..., 0]
            # update the curvature at the right node
            self.curvature_nodes[i+1, :] = np.matmul(
                transformation_matrix_right_node, dtheta_prime_right_node[..., None])[..., 0] + \
                np.matmul(incremental_rotation_tensor_right_node,
                          self.curvature_nodes[i+1, :][..., None])[..., 0]

    def __compute_internal_forces_and_moments(self, e, element_unknowns, element_orientations, 
                                              element_curvatures, location="Quads"):
        """
        Compute the internal forces and moments for an element based on the provided unknowns,
        orientations, and curvatures.

        Parameters:
            e: The element index.
            element_unknowns: The unknowns of the element.
            element_orientations: The orientations of the element.
            element_curvatures: The curvatures of the element.
            location: The location where the internal forces and moments should computed.
                If "Quads", the internal forces and moments are computed at the quadrature points of the element.
                If "Nodes", the internal forces and moments are computed at the nodes of the element.
        Returns:
            internal_forces: The computed internal forces for the element.
            internal_moments: The computed internal moments for the element.
        """
        if (location != "Quads" and location != "Nodes"):
            raise ValueError("Location must be either 'Quads' or 'Nodes'.")
        ################# internal forces #################
        rp = self._compute_element_dof_derivatives(
            e, element_unknowns, self.function_space.local_translational_dofs, location=location)
        if (location == "Quads"):
            # check if the element orientations are of size (Q, dim)
            if (element_orientations.shape != (self.function_space.Q, self.function_space.dim)):
                raise ValueError("Element orientations must be of shape (Q, dim)")
        elif (location == "Nodes"):
            # check if the element orientations are of size (npel, dim)
            if (element_orientations.shape != (self.function_space.npel, self.function_space.dim)):
                raise ValueError("Element orientations must be of shape (npel, dim)")
        element_orientations_tensor = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(element_orientations))
        E1 = np.zeros([self.function_space.Q if location == "Quads" else self.function_space.npel, 
                       self.function_space.dim])
        E1[:, 0] = 1.0
        element_strains = rp - \
            np.matmul(element_orientations_tensor, E1[..., None])
        # translation constitutive matrix
        C_F = np.zeros([self.function_space.Q if location == "Quads" else self.function_space.npel,
                       self.function_space.dim, self.function_space.dim])
        C_F[:, 0, 0] = self.material.E * self.material.A
        C_F[:, 1, 1] = self.material.G * self.material.A_red
        C_F[:, 2, 2] = self.material.G * self.material.A_red
        # internal forces
        C_F_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_F, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        internal_forces = np.matmul(C_F_transformed, element_strains)
        ################# internal moments #################
        if (location == "Quads"):
            # check if the element curvatures are of size (Q, dim)
            if (element_curvatures.shape != (self.function_space.Q, self.function_space.dim)):
                raise ValueError("Element curvatures must be of shape (Q, dim)")
        elif (location == "Nodes"):
            # check if the element curvatures are of size (npel, dim)
            if (element_curvatures.shape != (self.function_space.npel, self.function_space.dim)):
                raise ValueError("Element curvatures must be of shape (npel, dim)")
        # rotational constitutive matrix
        C_M = np.zeros([self.function_space.Q if location == "Quads" else self.function_space.npel,
                       self.function_space.dim, self.function_space.dim])
        C_M[:, 0, 0] = self.material.G * self.material.I_T
        C_M[:, 1, 1] = self.material.E * self.material.I
        C_M[:, 2, 2] = self.material.E * self.material.I_minor
        # NOTE: Since the beam is initially straight, there is no initial curvature!!!
        C_M_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_M, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        internal_moments = np.matmul(
            C_M_transformed, element_curvatures[..., None])
        return internal_forces, internal_moments

    def compute_element_internal_forces(self, e, element_unknowns, element_orientations, 
                                        element_curvatures):
        """
        Compute the internal forces for an element based on the provided unknowns, orientations,
        and curvatures.

        Parameters:
            e: The element index.
            element_unknowns: The unknowns of the element.
            element_orientations: The orientations of the element at each quadrature point.
            element_curvatures: The curvatures of the element at each quadrature point.
        Returns:
            element_internal_forces: The computed internal forces for the element.
        """
        element_internal_forces = np.zeros(
            [self.function_space.dof*self.function_space.npel, 1])
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        rp = self._compute_element_dof_derivatives(
            e, element_unknowns, self.function_space.local_translational_dofs)
        # compute the internal forces and moments
        internal_forces, internal_moments = self.__compute_internal_forces_and_moments(
            e, element_unknowns, element_orientations, element_curvatures)
        # assemble the internal forces
        internal_forces_integrand = np.matmul(
            np.transpose(Np, axes=(0, 2, 1)), internal_forces)
        element_internal_forces[self.function_space.local_translational_dofs] += \
            np.sum(internal_forces_integrand *
                   self.function_space.JxW, axis=0, keepdims=False)
        # assemble the internal moments
        internal_moments_integrand = np.matmul(
            np.transpose(Np, axes=(0, 2, 1)), internal_moments)
        internal_moments_integrand -= np.matmul(
            np.transpose(N, axes=(0, 2, 1)), np.cross(rp, internal_forces, axis=1))
        element_internal_forces[self.function_space.local_rotational_dofs] += \
            np.sum(internal_moments_integrand *
                   self.function_space.JxW, axis=0, keepdims=False)
        return element_internal_forces

    def __compute_element_material_stiffness(self, e, element_unknowns, element_orientations):
        """
        Compute the material stiffness matrix for an element based on the provided unknowns and orientations.

        Parameters:
            element_unknowns: The unknowns of the element.
            element_orientations: The orientations of the element at each quadrature point.
        Returns:
            element_material_stiffness: The computed material stiffness matrix for the element.
        """
        element_material_stiffness = np.zeros(
            [self.function_space.dof*self.function_space.npel, 
             self.function_space.dof*self.function_space.npel])
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        rp = self._compute_element_dof_derivatives(
            e, element_unknowns, self.function_space.local_translational_dofs)
        # translational and rotational constitutive matrices
        C_F = np.zeros(
            [self.function_space.Q, self.function_space.dim, self.function_space.dim])
        C_F[:, 0, 0] = self.material.E * self.material.A
        C_F[:, 1, 1] = self.material.G * self.material.A_red
        C_F[:, 2, 2] = self.material.G * self.material.A_red
        C_M = np.zeros(
            [self.function_space.Q, self.function_space.dim, self.function_space.dim])
        C_M[:, 0, 0] = self.material.G * self.material.I_T
        C_M[:, 1, 1] = self.material.E * self.material.I
        C_M[:, 2, 2] = self.material.E * self.material.I_minor
        # check if the element orientations are of size (Q, dim)
        if (element_orientations.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element orientations must be of shape (Q, dim)")
        # transformed constitutive matrices
        element_orientations_tensor = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(element_orientations))
        C_F_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_F, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        C_M_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_M, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        # force derivatives w.r.t the translational dofs
        df_dd = np.matmul(np.transpose(Np, axes=(0, 2, 1)),
                          np.matmul(C_F_transformed, Np))
        element_material_stiffness[np.ix_(self.function_space.local_translational_dofs,
                                          self.function_space.local_translational_dofs)] += \
            np.sum(df_dd*self.function_space.JxW, axis=0, keepdims=False)
        # force derivatives w.r.t the rotational dofs
        rp_skew_matrix = skew_symmetric_matrices(rp[..., 0])
        df_dtheta_pre_multiplier = np.matmul(C_F_transformed, rp_skew_matrix)
        df_dtheta = np.matmul(np.transpose(Np, axes=(0, 2, 1)),
                              np.matmul(df_dtheta_pre_multiplier, N))
        element_material_stiffness[np.ix_(self.function_space.local_translational_dofs,
                                          self.function_space.local_rotational_dofs)] += \
            np.sum(df_dtheta*self.function_space.JxW, axis=0, keepdims=False)
        # moment derivatives w.r.t the translational dofs
        dm_dd_pre_multiplier = np.matmul(rp_skew_matrix, C_F_transformed)
        dm_dd = -1.0 * np.matmul(np.transpose(N, axes=(0, 2, 1)),
                                 np.matmul(dm_dd_pre_multiplier, Np))
        element_material_stiffness[np.ix_(self.function_space.local_rotational_dofs,
                                          self.function_space.local_translational_dofs)] += \
            np.sum(dm_dd*self.function_space.JxW, axis=0, keepdims=False)
        # moment derivatives w.r.t the rotational dofs
        ########## term 1 ##########
        dm_dtheta_term1_pre_multiplier = -1.0 * \
            np.matmul(rp_skew_matrix, np.matmul(
                C_F_transformed, rp_skew_matrix))
        dm_dtheta_term1 = np.matmul(np.transpose(N, axes=(0, 2, 1)),
                                    np.matmul(dm_dtheta_term1_pre_multiplier, N))
        element_material_stiffness[np.ix_(self.function_space.local_rotational_dofs,
                                          self.function_space.local_rotational_dofs)] += \
            np.sum(dm_dtheta_term1*self.function_space.JxW, axis=0, keepdims=False)
        ########## term 2 ##########
        dm_dtheta_term2 = np.matmul(np.transpose(
            Np, axes=(0, 2, 1)), np.matmul(C_M_transformed, Np))
        element_material_stiffness[np.ix_(self.function_space.local_rotational_dofs,
                                          self.function_space.local_rotational_dofs)] += \
            np.sum(dm_dtheta_term2*self.function_space.JxW, axis=0, keepdims=False)
        return element_material_stiffness

    def __compute_element_geometric_stiffness(self, e, element_unknowns, element_orientations,
                                              element_curvatures):
        """
        Compute the geometric stiffness matrix for an element based on the provided unknowns,
        orientations, and curvatures.

        Parameters:
            e: The element index.
            element_unknowns: The unknowns of the element.
            element_orientations: The orientations of the element at each quadrature point.
            element_curvatures: The curvatures of the element at each quadrature point.
        Returns:
            element_geometric_stiffness: The computed geometric stiffness matrix for the element.
        """
        element_geometric_stiffness = np.zeros(
            [self.function_space.dof*self.function_space.npel,
             self.function_space.dof*self.function_space.npel])
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        rp = self._compute_element_dof_derivatives(
            e, element_unknowns, self.function_space.local_translational_dofs)
        # check if the element orientations and curvatures are of size (Q, dim)
        if (element_orientations.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element orientations must be of shape (Q, dim)")
        if (element_curvatures.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element curvatures must be of shape (Q, dim)")
        # compute the internal forces and moments
        internal_forces, internal_moments = self.__compute_internal_forces_and_moments(
            e, element_unknowns, element_orientations, element_curvatures)
        internal_forces_skew_matrix = skew_symmetric_matrices(
            internal_forces[..., 0])
        internal_moments_skew_matrix = skew_symmetric_matrices(
            internal_moments[..., 0])
        # force derivatives w.r.t the rotational dofs
        df_dtheta = -1.0 * \
            np.matmul(np.transpose(Np, axes=(0, 2, 1)),
                      np.matmul(internal_forces_skew_matrix, N))
        element_geometric_stiffness[np.ix_(self.function_space.local_translational_dofs,
                                           self.function_space.local_rotational_dofs)] += \
            np.sum(df_dtheta*self.function_space.JxW, axis=0, keepdims=False)
        # moment derivatives w.r.t the translational dofs
        dm_dd = np.matmul(np.transpose(N, axes=(0, 2, 1)),
                          np.matmul(internal_forces_skew_matrix, Np))
        element_geometric_stiffness[np.ix_(self.function_space.local_rotational_dofs,
                                           self.function_space.local_translational_dofs)] += \
            np.sum(dm_dd*self.function_space.JxW, axis=0, keepdims=False)
        # moment derivatives w.r.t the rotational dofs
        ########## term 1 ##########
        dm_dtheta_term1 = -1.0 * np.matmul(np.transpose(
            Np, axes=(0, 2, 1)), np.matmul(internal_moments_skew_matrix, N))
        element_geometric_stiffness[np.ix_(self.function_space.local_rotational_dofs,
                                           self.function_space.local_rotational_dofs)] += \
            np.sum(dm_dtheta_term1*self.function_space.JxW, axis=0, keepdims=False)
        ########## term 2 ##########
        internal_forces_rp_outer = internal_forces @ np.transpose(
            rp, axes=(0, 2, 1))
        internal_forces_rp_dot = np.matmul(
            np.transpose(internal_forces, (0, 2, 1)), rp)
        dm_dtheta_term2_pre_multiplier = internal_forces_rp_outer - \
            internal_forces_rp_dot * np.eye(3)[np.newaxis, :, :]
        dm_dtheta_term2 = np.matmul(np.transpose(N, axes=(0, 2, 1)),
                                    np.matmul(dm_dtheta_term2_pre_multiplier, N))
        element_geometric_stiffness[np.ix_(self.function_space.local_rotational_dofs,
                                           self.function_space.local_rotational_dofs)] += \
            np.sum(dm_dtheta_term2*self.function_space.JxW, axis=0, keepdims=False)
        return element_geometric_stiffness

    def compute_element_internal_stiffness(self, e, element_unknowns, element_orientations, 
                                           element_curvatures):
        """
        Compute the internal stiffness matrix for an element based on the provided unknowns,
        orientations, and curvatures.

        Parameters:
            e: The element index.
            element_unknowns: The unknowns of the element.
            element_orientations: The orientations of the element at each quadrature point.
            element_curvatures: The curvatures of the element at each quadrature point.
        Returns:
            element_internal_stiffness: The computed internal stiffness matrix for the element.
        """
        element_internal_stiffness = np.zeros(
            [self.function_space.dof*self.function_space.npel, 
             self.function_space.dof*self.function_space.npel])
        # compute the element material stiffness
        element_internal_stiffness += self.__compute_element_material_stiffness(
            e, element_unknowns, element_orientations)
        # compute the element geometric stiffness
        element_internal_stiffness += self.__compute_element_geometric_stiffness(
            e, element_unknowns, element_orientations, element_curvatures)
        return element_internal_stiffness

    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        """
        Compute the system residual based on the provided unknowns and element loads.

        Parameters:
            f: The force vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
            update_internal: A boolean indicating whether to update the internal variables.
        """
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten(
            )
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads == None):  # No element loads
                f[global_element_dofs] -= self.compute_element_internal_forces(
                    i, element_unknowns, self.orientation[i, :, :], self.curvature[i, :, :])

    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        """
        Compute the system stiffness matrix based on the provided unknowns and loads.

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
                  ] += self.compute_element_internal_stiffness(i, element_unknowns, 
                                                               self.orientation[i, :, :], 
                                                               self.curvature[i, :, :])

    # Function to compute the system nodal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads):
        """
        Compute the system nodal forces based on the provided unknowns and element loads.

        Parameters:
            f: The force vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
        """
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            # NOTE: Here we assume that there are only two nodes per element!!!
            global_element_dofs_left_node = global_element_dofs[0:dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            element_unknowns = system_unknowns[global_element_dofs]
            nodal_orientations = element_unknowns[self.function_space.local_rotational_dofs].reshape(
                -1, self.function_space.dim)
            nodal_curvatures = np.stack(
                [self.curvature_nodes[i, :], self.curvature_nodes[i+1, :]], axis=0)
            # compute the internal forces and moments at the nodes
            internal_forces, internal_moments = self.__compute_internal_forces_and_moments(
                i, element_unknowns, nodal_orientations, nodal_curvatures, location="Nodes")
            if (i == 0):  # only for the first element
                # assemble the internal forces at the left node
                f[global_element_dofs_left_node[0:int(
                    dofs/2)]] += internal_forces[0, ...]
                f[global_element_dofs_left_node[int(
                    dofs/2):dofs]] += internal_moments[0, ...]
            # assemble the internal forces at the right node
            f[global_element_dofs_right_node[0:int(
                dofs/2)]] += internal_forces[1, ...]
            f[global_element_dofs_right_node[int(
                dofs/2):dofs]] += internal_moments[1, ...]
