import numpy as np
import quaternion
from beamit.WeakForm.WeakForm import WeakForm
from beamit.WeakForm.Utils import SolutionUpdateType, skew_symmetric_matrices
from beamit import Material


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
            [self.function_space.E*self.function_space.npel, self.function_space.dim])

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
        psi_norm_safe = np.where(small_norm_idxs, 1.0e-10, psi_norm)
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

    @staticmethod
    def _compute_transformation_matrix_inverse(orientation):
        """
        Compute the inverse of the transformation matrix (T) which converts multiplicative updates to additive updates.

        The inverse transformation matrix is computed based on the current orientation of the beam.

        Parameters:
            orientation: The orientation in terms of rotation vectors. **orienation** is assumed to be of shape (n, 3) where n is the number of rotation vectors.
        Returns:
            T_inv: The inverse transformation matrix of size (n, 3, 3).
        """
        # check if the orientation is of size (n, 3)
        if (orientation.ndim != 2 or orientation.shape[1] != 3):
            raise ValueError(
                "Orientation must be of shape (n, 3), where n is the number of rotation vectors.")
        psi_norm = np.linalg.norm(orientation, axis=1)
        I = np.eye(3)[None, :, :]
        # avoid division by zero
        small_norm_idxs = psi_norm < 1.0e-10
        psi_norm_safe = np.where(small_norm_idxs, 1.0e-10, psi_norm)
        psi_norm_half = psi_norm_safe / 2.0
        tan_psi_norm_half = np.tan(psi_norm_half)
        # auxiliary terms with safe evaluation
        t1 = np.where(small_norm_idxs, 1.0, psi_norm_half / tan_psi_norm_half)
        t2 = -0.50
        t3 = np.where(small_norm_idxs, 1.0/12.0,
                      (1.0 - psi_norm_half / tan_psi_norm_half) / psi_norm_safe**2)
        t1 = t1[:, None, None]
        t3 = t3[:, None, None]
        S = skew_symmetric_matrices(orientation)
        outer = orientation[:, :, None] * orientation[:, None, :]
        return t1 * I + t2 * S + t3 * outer

    @staticmethod
    def _compute_R_matrix(orientation, orientation_derivative):
        """
        Compute the R matrix which is an auxiliary matrix in the residual and stiffness computations.

        Parameters:
            orientation: The orientation in terms of rotation vectors. **orienation** is assumed to be of shape (n, 3) where n is the number of rotation vectors.
            orientation_derivative: The derivative of the orientation which is also assumed to be of shape (n, 3).
        Returns:
            R: The R matrix of size (n, 3, 3).
        """
        # check if the orientation is of size (n, 3)
        if (orientation.ndim != 2 or orientation.shape[1] != 3):
            raise ValueError(
                "Orientation must be of shape (n, 3), where n is the number of rotation vectors.")
        # check if the orientation_derivative is of size (n, 3)
        if (orientation_derivative.ndim != 2 or orientation_derivative.shape[1] != 3):
            raise ValueError(
                "Orientation derivative must be of shape (n, 3), where n is the number of rotation vectors.")
        psi_norm = np.linalg.norm(orientation, axis=1)
        I = np.eye(3)[None, :, :]
        # avoid division by zero
        small_norm_idxs = psi_norm < 1.0e-10
        psi_norm_safe = np.where(small_norm_idxs, 1.0e-10, psi_norm)
        psi_norm_safe2 = psi_norm_safe**2
        psi_norm_safe3 = psi_norm_safe**3
        psi_norm_safe4 = psi_norm_safe**4
        psi_norm_safe5 = psi_norm_safe**5
        # coefficients c1 to c5
        c1 = (psi_norm_safe * np.cos(psi_norm_safe) -
              np.sin(psi_norm_safe)) / psi_norm_safe3
        c2 = (psi_norm_safe * np.sin(psi_norm_safe) + 2 *
              np.cos(psi_norm_safe) - 2) / psi_norm_safe4
        c3 = (3 * np.sin(psi_norm_safe) - 2 * psi_norm_safe -
              psi_norm_safe * np.cos(psi_norm_safe)) / psi_norm_safe5
        c4 = (1 - np.cos(psi_norm_safe)) / psi_norm_safe2
        c5 = (psi_norm_safe - np.sin(psi_norm_safe)) / psi_norm_safe3
        # apply limits for small angles
        c1 = np.where(small_norm_idxs, -1.0 / 3.0, c1)
        c2 = np.where(small_norm_idxs, -1.0 / 12.0, c2)
        c3 = np.where(small_norm_idxs, 1.0 / 20.0, c3)
        c4 = np.where(small_norm_idxs, 0.50, c4)
        c5 = np.where(small_norm_idxs, 1.0 / 6.0, c5)
        c1 = c1[:, None, None]
        c2 = c2[:, None, None]
        c3 = c3[:, None, None]
        c4 = c4[:, None, None]
        c5 = c5[:, None, None]
        # compute auxiliary terms
        dot = np.einsum('ni,ni->n', orientation, orientation_derivative)
        cross = np.cross(orientation, orientation_derivative, axis=1)
        outer_theta = orientation[:, :, None] * orientation[:, None, :]
        outer_thetas = orientation_derivative[:, :, None] * orientation[:, None, :]
        outer_theta_cross = cross[:, :, None] * orientation[:, None, :]
        outer_theta_dot = dot[:, None, None] * outer_theta
        skew_theta_s = skew_symmetric_matrices(orientation_derivative)
        identity = I.repeat(orientation.shape[0], axis=0)
        identity_scaled = dot[:, None, None] * identity
        outer_theta_thetas = orientation[:, :, None] * orientation_derivative[:, None, :]
        # compute the R matrix
        R = (
            c1 * outer_thetas +
            c2 * outer_theta_cross +
            c3 * outer_theta_dot -
            c4 * skew_theta_s +
            c5 * (identity_scaled + outer_theta_thetas)
        )
        return R

    def _update_element_internal_variables(self, e, system_unknowns_increment):
        """
        Update the internal variables of an element based on the current system unknowns increment.

        This method updates the orientation and curvature of the element at each quadrature point 
        based on the current system unknowns increment.

        Parameters:
            e: The index of the element.
            system_unknowns_increment: The increment in the system unknowns.
        """
        N = self.function_space.shape_functions
        global_element_dofs = self.function_space.global_connectivity[e:e+1].flatten()
        element_rotation_increment = system_unknowns_increment[
            global_element_dofs][self.function_space.local_rotational_dofs]
        # compute the "additive" rotation increment and its derivative
        dtheta = np.matmul(N, element_rotation_increment)[..., 0]
        dtheta_prime = self._compute_element_dof_derivatives(
            e, system_unknowns_increment[global_element_dofs], 
            self.function_space.local_rotational_dofs)[..., 0]
        ############# update the orientation #############
        current_orientation = self.orientation[e, :, :]
        # NOTE: Careful with the order of multiplication here since quaternion multiplication 
        # is not commutative!
        updated_orientation_quats = \
            quaternion.from_rotation_vector(dtheta) * quaternion.from_rotation_vector(
                current_orientation)
        # convert the updated orientation quaternions to rotation vectors
        updated_orientation = quaternion.as_rotation_vector(updated_orientation_quats)
        self.orientation[e, :, :] = updated_orientation
        ############# update the curvature #############
        # compute the transformation matrix
        transformation_matrix = self._compute_transformation_matrix(dtheta)
        incremental_rotation_tensor = \
            quaternion.as_rotation_matrix(
                quaternion.from_rotation_vector(dtheta))
        current_curvature = self.curvature[e, :, :]
        self.curvature[e, :, :] = np.matmul(
            transformation_matrix, dtheta_prime[..., None])[..., 0] + \
            np.matmul(incremental_rotation_tensor,
                        current_curvature[..., None])[..., 0]

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

    def update_rotational_solution(self, system_unknowns, solution_increment):
        """
        Update the rotational degrees of freedom in the solution based on the solution increment.

        Parameters:
            system_unknowns: The unknowns of the system.
            solution_increment: The increment in the solution.
        """
        # get the rotational dof indices
        dof = self.function_space.dof
        num_rot_dofs = (int)(self.function_space.local_rotational_dofs.size /
                             self.function_space.npel)
        rotational_dof_indices = np.concatenate(
            [self.function_space.local_rotational_dofs[0:num_rot_dofs] +
             dof*i for i in range(self.function_space.N)])
        rotation_solution_vectors = \
            system_unknowns[rotational_dof_indices].reshape([-1, 3])
        rotation_increment_vectors = \
            solution_increment[rotational_dof_indices].reshape([-1, 3])
        # convert the rotation vectors to quaternions
        rotation_solution_quats = quaternion.from_rotation_vector(rotation_solution_vectors)
        rotation_increment_quats = quaternion.from_rotation_vector(rotation_increment_vectors)
        # update the rotations using quaternion multiplication
        # NOTE: Here, we need to be careful with the order of multiplication since quaternion
        # multiplication is not commutative!!!
        updated_rotation_quats = rotation_increment_quats * rotation_solution_quats
        # convert the updated quaternions back to rotation vectors
        updated_rotation_vectors = quaternion.as_rotation_vector(updated_rotation_quats)
        # update the solution vector with the updated rotation vectors
        system_unknowns[rotational_dof_indices] = updated_rotation_vectors.reshape([-1, 1])

    def update_internal_variables(self, system_unknowns_increment):
        """
        Update the internal variables in the weak form based on the increment in the system unknowns.
        
        The orientation and curvature of the beam at each quadrature point are updated based on 
        the current system unknowns increment.

        Parameters:
            system_unknowns_increment: The increment in the system unknowns.
        """
        # update the orientation and curvature at quadrature points
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            self._update_element_internal_variables(
                i, system_unknowns_increment)
        ########## update the curvature at the nodes ##########
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
            # incremental transformation matrices
            incremental_transformation_matrix_left_node = self._compute_transformation_matrix(
                dtheta_left_node[None, ...])[0, ...]
            incremental_transformation_matrix_right_node = self._compute_transformation_matrix(
                dtheta_right_node[None, ...])[0, ...]
            incremental_rotation_tensor_left_node = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta_left_node))
            incremental_rotation_tensor_right_node = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta_right_node))
            # update the curvature at the left node
            self.curvature_nodes[2*i, :] = np.matmul(
                incremental_transformation_matrix_left_node, dtheta_prime_left_node[..., None])[..., 0] + \
                np.matmul(incremental_rotation_tensor_left_node,
                            self.curvature_nodes[2*i, :][..., None])[..., 0]
            # update the curvature at the right node
            self.curvature_nodes[2*i+1, :] = np.matmul(
                incremental_transformation_matrix_right_node, dtheta_prime_right_node[..., None])[..., 0] + \
                np.matmul(incremental_rotation_tensor_right_node,
                            self.curvature_nodes[2*i+1, :][..., None])[..., 0]

    def _compute_internal_forces_and_moments(self, e, element_unknowns, element_orientations, 
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
        internal_forces, internal_moments = self._compute_internal_forces_and_moments(
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

    def _compute_element_material_stiffness(self, e, element_unknowns, element_orientations):
        """
        Compute the material stiffness matrix for an element based on the provided unknowns and orientations.

        NOTE: This method computes the material stiffness matrix for the element only when we consider
        the multiplicative updates of the rotational degrees of freedom. But currently we are using 
        the additive updates, so do not use this method!!!

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

    def _compute_element_geometric_stiffness(self, e, element_unknowns, element_orientations,
                                              element_curvatures):
        """
        Compute the geometric stiffness matrix for an element based on the provided unknowns,
        orientations, and curvatures.

        NOTE: This method computes the geometric stiffness matrix for the element only when we consider
        the multiplicative updates of the rotational degrees of freedom. But currently we are using 
        the additive updates, so do not use this method!!!

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
        internal_forces, internal_moments = self._compute_internal_forces_and_moments(
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

    def __compute_element_numerical_stiffness(self, e, system_unknowns):
        """
        Compute the numerical stiffness matrix for an element based on the provided unknowns.

        Parameters:
            e: The element index.
            system_unknowns: The unknowns of the system.
        Returns:
            element_numerical_stiffness: The computed numerical stiffness matrix for the element.
        """
        perturbation_factor = 1.0e-05
        np.random.seed(1234 + e)  # for reproducibility
        std_dev_system_unknowns = 0.01 * \
            np.maximum(np.abs(system_unknowns), 1.0e-03)
        perturbation_magnitudes = perturbation_factor * np.abs(
            np.random.normal(loc=system_unknowns, scale=std_dev_system_unknowns))
        global_element_dofs = self.function_space.global_connectivity[e:e+1].flatten()
        # perturb the element dofs individually to compute the numerical stiffness matrix
        element_numerical_stiffness = np.zeros(
            [global_element_dofs.shape[0], global_element_dofs.shape[0]])
        perturbed_system_unknowns = system_unknowns.copy()
        perturbed_solution_increments = np.zeros_like(system_unknowns)
        for i in range(0, global_element_dofs.shape[0]):
            # reset the perturbed increments
            perturbed_solution_increments.fill(0.0)
            # perturb the i-th dof
            perturbed_solution_increments[global_element_dofs[i]] = \
                perturbation_magnitudes[global_element_dofs[i]]
            ####### positively perturb the unknowns #######
            perturbed_system_unknowns += \
                perturbed_solution_increments
            # update the internal variables of the element
            self._update_element_internal_variables(
                e, perturbed_solution_increments)
            # compute the element internal forces for the positive perturbation
            element_internal_forces_positive_perturbation = self.compute_element_internal_forces(
                e, perturbed_system_unknowns[global_element_dofs], 
                self.orientation[e, :, :], self.curvature[e, :, :])
            ####### negatively perturb the unknowns #######
            perturbed_system_unknowns -= \
                2.0 * perturbed_solution_increments
            # update the internal variables of the element
            self._update_element_internal_variables(
                e, -2.0 * perturbed_solution_increments)
            # compute the element internal forces for the negative perturbation
            element_internal_forces_negative_perturbation = self.compute_element_internal_forces(
                e, perturbed_system_unknowns[global_element_dofs],
                self.orientation[e, :, :], self.curvature[e, :, :])
            # compute the element numerical stiffness matrix
            element_numerical_stiffness[:, i:i+1] += \
                (element_internal_forces_positive_perturbation - \
                 element_internal_forces_negative_perturbation) / \
                (2.0 * perturbation_magnitudes[global_element_dofs[i]])
            ####### reset the unknowns and internal variables #######
            perturbed_system_unknowns += \
                perturbed_solution_increments
            self._update_element_internal_variables(
                e, perturbed_solution_increments)
        return element_numerical_stiffness

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
        element_internal_stiffness += self._compute_element_material_stiffness(
            e, element_unknowns, element_orientations)
        # compute the element geometric stiffness
        element_internal_stiffness += self._compute_element_geometric_stiffness(
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
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
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
            if (element_loads == None):  # No element loads
                element_unknowns = system_unknowns[global_element_dofs]
                A[np.ix_(global_element_dofs, global_element_dofs)] += \
                    self.compute_element_internal_stiffness(i, element_unknowns,
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
                [self.curvature_nodes[2*i, :], self.curvature_nodes[2*i+1, :]], axis=0)
            # compute the internal forces and moments at the nodes
            internal_forces, internal_moments = self._compute_internal_forces_and_moments(
                i, element_unknowns, nodal_orientations, nodal_curvatures, location="Nodes")
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

    def __compute_element_rotational_mass(self, e, dt, element_angular_velocities):
        """
        Compute the rotational mass matrix of an element based on the provided unknowns

        Parameters:
            e: The element index.
            dt: The time step size.
            element_angular_velocities: The angular velocities of the element.
        Returns:
            M_el_rot: The rotational mass matrix of the element.
            r_el_rot: The rotational inertia vector of the element.
        """
        # get the shape functions
        N = self.function_space.shape_functions
        Nt = np.transpose(N, axes=(0, 2, 1))
        # inertia tensor
        C_rho = np.zeros(
            [self.function_space.Q, self.function_space.dim, self.function_space.dim])
        C_rho[:, 0, 0] = self.material.rho * self.material.I_T
        C_rho[:, 1, 1] = self.material.rho * self.material.I
        C_rho[:, 2, 2] = self.material.rho * self.material.I_minor
        # transformed inertia tensor
        element_orientations = self.orientation[e, :, :]
        element_orientations_tensor = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(element_orientations))
        C_rho_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_rho, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        ########## rotational mass ##########
        # first part of the rotational mass integrand
        rotational_mass_integrand = np.matmul(Nt, np.matmul(
            C_rho_transformed, N))
        # second part of the rotational mass integrand
        rot_mass_second_part_aux = np.matmul(
            C_rho_transformed, np.matmul(N, element_angular_velocities))
        rot_mass_second_part_aux = skew_symmetric_matrices(
            rot_mass_second_part_aux[..., 0])
        rotational_mass_integrand -= 0.50 * dt * np.matmul(Nt, np.matmul(
            rot_mass_second_part_aux, N))
        # third part of the rotational mass integrand
        rot_mass_third_part_aux = np.matmul(
            N, element_angular_velocities)
        rot_mass_third_part_aux = skew_symmetric_matrices(
            rot_mass_third_part_aux[..., 0])
        rot_mass_third_part_aux = np.matmul(
            rot_mass_third_part_aux, C_rho_transformed)
        rotational_mass_integrand += 0.50 * dt * np.matmul(Nt, np.matmul(
            rot_mass_third_part_aux, N))
        # NOTE: Here we are not adding the term to the mass matrix which involves the angular
        # acceleration as that term will be small due to (dt/2) extra prefactor and the initial
        # angular acceleration at the start of a time step is zero anyways. We probably have to
        # implement that term if we are using this mass matrix within the context of an implicit
        # time integration scheme.
        M_el_rot = np.sum(rotational_mass_integrand *
                          self.function_space.JxW, axis=0, keepdims=False)
        ########## rotational inertia ##########
        rot_inertia_aux1 = np.matmul(N, element_angular_velocities)
        rot_inertia_aux2 = skew_symmetric_matrices(rot_inertia_aux1[..., 0])
        rot_inertia_aux2 = np.matmul(rot_inertia_aux2, np.matmul(
            C_rho_transformed, rot_inertia_aux1))
        rotational_inertia_integrand = np.matmul(Nt, rot_inertia_aux2)
        f_el_rot = np.sum(rotational_inertia_integrand *
                          self.function_space.JxW, axis=0, keepdims=False)
        return M_el_rot, f_el_rot

    def compute_system_mass(self, M, lump=True, **kwargs):
        """
        Compute the system mass matrix.

        Parameters:
            M: The mass matrix to be assembled.
            lump: If True, apply lumping to the mass matrix.
            **kwargs: Optional keyword arguments.
        """
        # get the shape functions
        N = self.function_space.shape_functions
        Nt = np.transpose(N, axes=(0, 2, 1))
        # element translational mass matrix
        translational_mass_integrand = self.material.rho*self.material.A * \
            np.matmul(Nt, N)
        M_el_trans = np.sum(translational_mass_integrand *
                            self.function_space.JxW, axis=0, keepdims=False)
        # get additional parameters from kwargs
        dt = kwargs.get("dt", 0.0)
        system_velocities = kwargs.get("system_velocities", np.zeros(
            [self.function_space.N*self.function_space.dof, 1]))
        residual_vector = kwargs.get("residual_vector", None)
        # container to store element mass matrix
        M_el = np.zeros(
                [self.function_space.dof*self.function_space.npel,
                 self.function_space.dof*self.function_space.npel])
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            # compute the element rotational mass matrix and rotational inertia vector
            M_el_rot, f_el_rot = self.__compute_element_rotational_mass(
                i, dt, system_velocities[global_element_dofs[self.function_space.local_rotational_dofs]])
            # reinitialize element mass matrix and add contributions
            M_el.fill(0.0)
            M_el[np.ix_(self.function_space.local_translational_dofs,
                         self.function_space.local_translational_dofs)] += M_el_trans
            M_el[np.ix_(self.function_space.local_rotational_dofs,
                         self.function_space.local_rotational_dofs)] += M_el_rot
            # apply lumping if needed (based on row sum technique)
            if (lump):
                M_el = np.diag(np.sum(M_el, axis=1))
            # assemble the element mass to the global mass matrix
            M[np.ix_(global_element_dofs, global_element_dofs)] += M_el
            # add the element rotational inertia contribution to the residual vector
            if (residual_vector is not None):
                residual_vector[global_element_dofs[self.function_space.local_rotational_dofs]
                                ] -= f_el_rot

class ShearFlexibleGeometricallyExactWeakFormDG(ShearFlexibleGeometricallyExactWeakFormCG):

    def __init__(self, function_space, material, betaP, betaT):
        """
        Initialize the ShearFlexibleGeometricallyExactWeakFormDG class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
            betaP: The penalty parameter for the position jump.
            betaT: The penalty parameter for the rotation jump.
        """
        # invoke the parent (ShearFlexibleGeometricallyExactWeakFormCG) class
        ShearFlexibleGeometricallyExactWeakFormCG.__init__(
            self, function_space, material)
        # DG position jump penalty parameter
        self.betaP = betaP
        # DG rotation jump penalty parameter
        self.betaT = betaT
        # container to store dof jumps at the element boundaries
        self.dof_jumps_boundaries = np.zeros(
            [self.function_space.E, 2*self.function_space.dof])
        # store internal variables (of CZM) at all the interfaces
        # first column = binary parameter to indicate damage status at the interface
        # (0.0 -> damage 'not' initiated, 1.0 -> damage initiated)
        # second column = binary parameter to enable/disable DG and CZM terms in the interface jump forces
        # (0.0 -> DG terms are active, 1.0 -> CZM terms are active)
        # third column = maximum effective separation at an interface in the entire loading history
        self.internal_variables = np.zeros([self.function_space.E-1, 3])

    def __get_dof_jumps_at_element_boundaries(self, e, system_dof_values):
        """
        Get the dof jumps at the boundaries of the element.

        This method only works if the element are arranged in a linear fashion i.e. the first element is the left most element and the last element is the right most element and the other elements are sequentially arranged in between. In any other case, the method will not work as expected.

        Parameters:
            e: The index of the element.
            system_dof_values: The dof values of the system.
        Returns:
            dof_jumps_boundaries: The dof jumps at the element boundaries.
        """
        # compute the dof jumps at the element boundaries
        dofs = self.function_space.dof
        dof_jumps_boundaries = np.zeros([2*dofs, 1])
        global_element_dofs = self.function_space.global_connectivity[e:e+1].flatten()
        element_dof_values = system_dof_values[global_element_dofs]
        if (e == 0):  # for left most element
            right_element_dofs = self.function_space.global_connectivity[e+1:e+2].flatten()
            right_element_dof_values = system_dof_values[right_element_dofs]
            dof_jumps_boundaries[dofs:] = right_element_dof_values[0:dofs] - \
                element_dof_values[dofs:]
        elif (e == self.function_space.E-1):  # for right most element
            left_element_dofs = self.function_space.global_connectivity[e-1:e].flatten()
            left_element_dof_values = system_dof_values[left_element_dofs]
            dof_jumps_boundaries[0:dofs] = element_dof_values[0:dofs] - \
                left_element_dof_values[dofs:]
        else:  # for intermediate elements
            left_element_dofs = self.function_space.global_connectivity[e-1:e].flatten()
            left_element_dof_values = system_dof_values[left_element_dofs]
            right_element_dofs = self.function_space.global_connectivity[e+1:e+2].flatten()
            right_element_dof_values = system_dof_values[right_element_dofs]
            dof_jumps_boundaries[0:dofs] = element_dof_values[0:dofs] - \
                left_element_dof_values[dofs:]
            dof_jumps_boundaries[dofs:] = right_element_dof_values[0:dofs] - \
                element_dof_values[dofs:]
        return dof_jumps_boundaries

    def _update_element_internal_variables(self, e, system_unknowns_increment):
        """
        Update the internal variables of an element based on the current system unknowns increment.

        This method updates the orientation and curvature of the element at each quadrature point and
        the rotational dof jumps at the element boundaries based on the current system unknowns increment.

        Parameters:
            e: The index of the element.
            system_unknowns_increment: The increment in the system unknowns.
        """
        # update the orientation and curvature at quadrature points
        super()._update_element_internal_variables(
            e, system_unknowns_increment)
        ############# update the boundary dof jumps #############
        # NOTE: Here we only update the rotational dof jumps. The translational dof jumps are
        # updated through a different method in stiffness and residual assembly methods.
        element_dof_increment_jumps_boundaries = self.__get_dof_jumps_at_element_boundaries(
            e, system_unknowns_increment)
        self.dof_jumps_boundaries[e, self.function_space.local_rotational_dofs] = \
            element_dof_increment_jumps_boundaries[self.function_space.local_rotational_dofs, 0]

    def __update_element_position_jumps(self, e, system_unknowns):
        """
        Update the position jumps at the element boundaries based on the current system unknowns.

        This method computes the position jumps at the element boundaries and updates the 
        self.dof_jumps_boundaries container.
        
        Parameters:
            e: The index of the element.
            system_unknowns: The current system unknowns.
        """
        element_dof_jumps_boundaries = self.__get_dof_jumps_at_element_boundaries(
            e, system_unknowns)
        self.dof_jumps_boundaries[e, self.function_space.local_translational_dofs] = \
            element_dof_jumps_boundaries[self.function_space.local_translational_dofs, 0]

    def __update_system_position_jumps(self, system_unknowns):
        """
        Update the system position jumps based on the current system unknowns.

        This method computes the position jumps at the element boundaries and updates the 
        self.dof_jumps_boundaries container.
        
        Parameters:
            system_unknowns: The current system unknowns.
        """
        for i in range(0, self.function_space.E):
            self.__update_element_position_jumps(i, system_unknowns)

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
            lifting_shapes = self.function_space.lifting_shape_functions * \
                (1.0/self.function_space.jacobian)
        elif (location == "Nodes"):
            # NOTE: Here we are assuming that there are only two nodes per element!!!
            _, Nxi_left_node = self.function_space.compute_shapes(-1.0)
            Np_left_node = Nxi_left_node*(1.0/self.function_space.jacobian)
            _, Nxi_right_node = self.function_space.compute_shapes(1.0)
            Np_right_node = Nxi_right_node*(1.0/self.function_space.jacobian)
            Np = np.stack([Np_left_node, Np_right_node], axis=0)
            lifting_shapes_left_node = self.function_space.compute_lifting_shapes(-1.0) * \
                (1.0/self.function_space.jacobian)
            lifting_shapes_right_node = self.function_space.compute_lifting_shapes(1.0) * \
                (1.0/self.function_space.jacobian)
            lifting_shapes = np.stack(
                [lifting_shapes_left_node, lifting_shapes_right_node], axis=0)
        return np.matmul(Np, element_dof_values[local_dof_indices]) + \
            np.matmul(lifting_shapes, (self.dof_jumps_boundaries[e:e+1, local_dof_indices].T))

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
        # container to store the extended element internal forces
        # NOTE: The container is extended to account for the coupling between the current element 
        # and its neighbors due to the lifting terms.
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        # get the lifting shape functions at the quadrature points
        lifting_shapes = self.function_space.lifting_shape_functions * \
                (1.0/self.function_space.jacobian)
        element_internal_forces = np.zeros(
            [dofspel+dofs, 1]) if (e == 0 or e == self.function_space.E-1) else \
            np.zeros([2*dofspel, 1])
        # compute the element internal forces from the parent class
        element_bulk_internal_forces = super().compute_element_internal_forces(e, element_unknowns, 
                                                                               element_orientations, 
                                                                               element_curvatures)
        # compute the internal forces and moments for this element
        internal_forces, internal_moments = self._compute_internal_forces_and_moments(
            e, element_unknowns, self.orientation[e, :, :], self.curvature[e, :, :])
        # compute the element lifting forces and moments
        lifting_forces_integrand = np.matmul(np.transpose(
            lifting_shapes, axes=(0, 2, 1)), internal_forces)
        lifting_moments_integrand = np.matmul(np.transpose(
            lifting_shapes, axes=(0, 2, 1)), internal_moments)
        element_lifting_forces = np.sum(
            lifting_forces_integrand*self.function_space.JxW, axis=0, keepdims=False)
        element_lifting_moments = np.sum(
            lifting_moments_integrand*self.function_space.JxW, axis=0, keepdims=False)
        # assemble the internal forces
        if (e == 0):  # for the left most element
            element_internal_forces[0:dofspel] += element_bulk_internal_forces
            # add lifting forces
            element_internal_forces[dofs:dofs+int(dofs/2)] -= \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_forces[int(dofs/2):dofs]
            element_internal_forces[dofspel:dofspel+int(dofs/2)] += \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_forces[int(dofs/2):dofs]
            # add lifting moments
            element_internal_forces[dofs+int(dofs/2):dofspel] -= \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_moments[int(dofs/2):dofs]
            element_internal_forces[dofspel+int(dofs/2):dofspel+dofs] += \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_moments[int(dofs/2):dofs]
        elif (e == self.function_space.E-1):  # for the right most element
            element_internal_forces[dofs:(dofs+dofspel)] += element_bulk_internal_forces
            # add lifting forces
            element_internal_forces[0:int(dofs/2)] -= \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_forces[0:int(dofs/2)]
            element_internal_forces[dofs:dofs+int(dofs/2)] += \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_forces[0:int(dofs/2)]
            # add lifting moments
            element_internal_forces[int(dofs/2):dofs] -= \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_moments[0:int(dofs/2)]
            element_internal_forces[dofs+int(dofs/2):dofspel] += \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_moments[0:int(dofs/2)]
        else:  # for the intermediate elements
            element_internal_forces[dofs:(dofs+dofspel)] += element_bulk_internal_forces
            # add lifting forces
            element_internal_forces[0:int(dofs/2)] -= \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_forces[0:int(dofs/2)]
            element_internal_forces[dofs:dofs+int(dofs/2)] += \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_forces[0:int(dofs/2)]
            element_internal_forces[dofspel:dofspel+int(dofs/2)] -= \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_forces[int(dofs/2):dofs]
            element_internal_forces[dofspel+dofs:dofspel+dofs+int(dofs/2)] += \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_forces[int(dofs/2):dofs]
            # add lifting moments
            element_internal_forces[int(dofs/2):dofs] -= \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_moments[0:int(dofs/2)]
            element_internal_forces[dofs+int(dofs/2):dofspel] += \
                (1.0 - self.internal_variables[e-1:e, 1:2]) * \
                    element_lifting_moments[0:int(dofs/2)]
            element_internal_forces[dofspel+int(dofs/2):dofspel+dofs] -= \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_moments[int(dofs/2):dofs]
            element_internal_forces[dofspel+dofs+int(dofs/2):2*dofspel] += \
                (1.0 - self.internal_variables[e:e+1, 1:2]) * \
                    element_lifting_moments[int(dofs/2):dofs]
        return element_internal_forces

    def __compute_CZM_interface_forces(self, r_left_interface, r_right_interface,
                                       psi_left_interface, psi_right_interface, 
                                       forces_left_interface, forces_right_interface, 
                                       moments_left_interface, moments_right_interface, 
                                       element_internal_variables, update_internal):
        """
        Compute the cohesive zone model (CZM) interface forces and moments.

        Parameters:
            r_left_interface: The position vector at the left interface.
            r_right_interface: The position vector at the right interface.
            psi_left_interface: The orientation vector at the left interface.
            psi_right_interface: The orientation vector at the right interface.
            forces_left_interface: The forces at the left interface.
            forces_right_interface: The forces at the right interface.
            moments_left_interface: The moments at the left interface.
            moments_right_interface: The moments at the right interface.
            element_internal_variables: The internal variables at the interface.
            update_internal: If True, update the internal variables in the weak form.
        Returns:
            cohesive_forces: The cohesive forces at the interface (if applicable).
            cohesive_moments: The cohesive moments at the interface (if applicable).
        """
        # initialize cohesive forces and moments
        cohesive_forces = np.zeros([self.function_space.dim, 1])
        cohesive_moments = np.zeros([self.function_space.dim, 1])
        # perform CZM checks and calculations in the case of a cohesive interface material
        if ((isinstance(self.material, (Material.ShearFlexibleCohesiveInterfaceMaterial))) and update_internal):
            # just after damage initiation or damage not yet initiated
            if (element_internal_variables[0:1, 2:3] == 0.0):
                # just after damage initiation
                if (element_internal_variables[0:1, 0:1] == 1.0):
                    # effective separation at the interface
                    delta = self.material.compute_effective_separation(r_left_interface,
                                                                       r_right_interface,
                                                                       psi_left_interface,
                                                                       psi_right_interface)
                    if (delta == 0.0):  # fall back to DG terms
                        element_internal_variables[0:1, 1:2] = 0.0
                    else:  # perform CZM calculations
                        element_internal_variables[0:1, 1:2] = 1.0
                        # evaluate cohesive forces and moments according to the TSL
                        cohesive_forces, cohesive_moments = \
                            self.material.compute_cohesive_forces_and_moments(
                                r_left_interface, r_right_interface, psi_left_interface, 
                                psi_right_interface, delta_max=element_internal_variables[0:1, 2:3])
                        if (update_internal):
                            # update the maximum effective separation
                            new_delta_max = self.material.compute_effective_maximum_separation(
                                delta, delta_max=element_internal_variables[0:1, 2:3])
                            element_internal_variables[0:1, 2:3] = new_delta_max
                # damage not yet initiated
                # evaluate the damage initiation criterion
                elif (self.material.evaluate_damage_initiation_criterion(
                        psi_left_interface, psi_right_interface, forces_left_interface, 
                        forces_right_interface, moments_left_interface, moments_right_interface)):
                    # damage just initiated at the interface
                    element_internal_variables[0:1, 0:1] = 1.0
                    # keep the DG terms active immediately after damage initiation (since delta = 0.0)
                    element_internal_variables[0:1, 1:2] = 0.0
                else:  # no damage at the interface
                    element_internal_variables[0:1, 1:2] = 0.0
            # damage already initiated at the interface (loading | unloading | damage after recontact)
            else:
                # effective separation at the interface
                delta = self.material.compute_effective_separation(r_left_interface, 
                                                                   r_right_interface, 
                                                                   psi_left_interface, 
                                                                   psi_right_interface)
                if (delta == 0.0):  # fall back to DG terms
                    element_internal_variables[0:1, 1:2] = 0.0
                else:  # perform CZM calculations
                    element_internal_variables[0:1, 1:2] = 1.0
                    # evaluate cohesive forces and moments according to the TSL
                    cohesive_forces, cohesive_moments = \
                        self.material.compute_cohesive_forces_and_moments(
                            r_left_interface, r_right_interface, psi_left_interface, 
                            psi_right_interface, delta_max=element_internal_variables[0:1, 2:3])
                    if (update_internal):
                        # update the maximum effective separation
                        new_delta_max = self.material.compute_effective_maximum_separation(
                            delta, delta_max=element_internal_variables[0:1, 2:3])
                        element_internal_variables[0:1, 2:3] = new_delta_max
        return cohesive_forces, cohesive_moments

    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        """
        Compute the system residual based on the provided unknowns and element loads.

        Parameters:
            f: The force vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
            update_internal: A boolean indicating whether to update the internal variables.
        """
        # update the position jumps at the element boundaries before computing the residual
        # NOTE: This is necessary to ensure that the residual is computed with the correct 
        # position jumps.
        self.__update_system_position_jumps(system_unknowns)
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        # loop over the elements
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads == None):  # No element loads
                ###### assemble the element internal forces ######
                if (i == 0):  # for the left most element
                    right_element_dofs_left_node = \
                        self.function_space.global_connectivity[i+1:i+2].flatten()[0:dofs]
                    extended_element_dofs = np.append(
                        global_element_dofs, right_element_dofs_left_node)
                elif (i == self.function_space.E-1):  # for the right most element
                    left_element_dofs_right_node = \
                        self.function_space.global_connectivity[i-1:i].flatten()[dofs:dofspel]
                    extended_element_dofs = np.append(
                        left_element_dofs_right_node, global_element_dofs)
                else:  # for the intermediate elements
                    left_element_dofs_right_node = \
                        self.function_space.global_connectivity[i-1:i].flatten()[dofs:dofspel]
                    right_element_dofs_left_node = \
                        self.function_space.global_connectivity[i+1:i+2].flatten()[0:dofs]
                    extended_element_dofs = np.append(
                        left_element_dofs_right_node, np.append(
                            global_element_dofs, right_element_dofs_left_node))
                f[extended_element_dofs] -= self.compute_element_internal_forces(
                    i, element_unknowns, self.orientation[i, :, :], self.curvature[i, :, :])
        # assemble the interface terms
        N_left_interface, _ = self.function_space.compute_shapes(1.0)
        N_right_interface, _ = self.function_space.compute_shapes(-1.0)
        # loop over the interface elements
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            # compute the forces and moments for the left and right elements
            nodal_orientations_left = element_unknowns_left[
                self.function_space.local_rotational_dofs].reshape(-1, self.function_space.dim)
            nodal_orientations_right = element_unknowns_right[
                self.function_space.local_rotational_dofs].reshape(-1, self.function_space.dim)
            nodal_curvatures_left = np.stack(
                [self.curvature_nodes[2*i, :], self.curvature_nodes[2*i+1, :]], axis=0)
            nodal_curvatures_right = np.stack(
                [self.curvature_nodes[2*(i+1), :], self.curvature_nodes[2*(i+1)+1, :]], axis=0)
            forces_left, moments_left = self._compute_internal_forces_and_moments(
                i, element_unknowns_left, nodal_orientations_left, 
                nodal_curvatures_left, location="Nodes")
            forces_right, moments_right = self._compute_internal_forces_and_moments(
                i+1, element_unknowns_right, nodal_orientations_right, 
                nodal_curvatures_right, location="Nodes")
            # get the dofs at the left and right sides of the interface
            r_left_interface = element_unknowns_left[self.function_space.local_translational_dofs[
                self.function_space.dim:]]
            r_right_interface = element_unknowns_right[self.function_space.local_translational_dofs[
                0:self.function_space.dim]]
            psi_left_interface = element_unknowns_left[self.function_space.local_rotational_dofs[
                self.function_space.dim:]]
            psi_right_interface = element_unknowns_right[self.function_space.local_rotational_dofs[
                0:self.function_space.dim]]
            # penalty terms
            penalty_forces = self.betaP * \
                ((self.material.E*self.material.A) / self.function_space.elL) * \
                (self.dof_jumps_boundaries[i:i+1, self.function_space.local_translational_dofs].T)[
                    self.function_space.dim:]
            penalty_moments = self.betaT * \
                ((self.material.E*self.material.I) / self.function_space.elL) * \
                (self.dof_jumps_boundaries[i:i+1, self.function_space.local_rotational_dofs].T)[
                    self.function_space.dim:]
            f[global_element_dofs_left[self.function_space.local_translational_dofs]] += \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * np.matmul(
                    np.transpose(N_left_interface), penalty_forces)
            f[global_element_dofs_left[self.function_space.local_rotational_dofs]] += \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * np.matmul(
                    np.transpose(N_left_interface), penalty_moments)
            f[global_element_dofs_right[self.function_space.local_translational_dofs]] -= \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * np.matmul(
                    np.transpose(N_right_interface), penalty_forces)
            f[global_element_dofs_right[self.function_space.local_rotational_dofs]] -= \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * np.matmul(
                    np.transpose(N_right_interface), penalty_moments)
            # CZM terms
            cohesive_forces, cohesive_moments = self.__compute_CZM_interface_forces(
                r_left_interface, r_right_interface, psi_left_interface, psi_right_interface,
                forces_left[1, ...], forces_right[0, ...], 
                moments_left[1, ...], moments_right[0, ...], 
                self.internal_variables[i:i+1, :], update_internal)
            f[global_element_dofs_left[self.function_space.local_translational_dofs]] += \
                self.internal_variables[i:i+1, 1:2] * np.matmul(
                    np.transpose(N_left_interface), cohesive_forces)
            f[global_element_dofs_left[self.function_space.local_rotational_dofs]] += \
                self.internal_variables[i:i+1, 1:2] * np.matmul(
                    np.transpose(N_left_interface), cohesive_moments)
            f[global_element_dofs_right[self.function_space.local_translational_dofs]] -= \
                self.internal_variables[i:i+1, 1:2] * np.matmul(
                    np.transpose(N_right_interface), cohesive_forces)
            f[global_element_dofs_right[self.function_space.local_rotational_dofs]] -= \
                self.internal_variables[i:i+1, 1:2] * np.matmul(
                    np.transpose(N_right_interface), cohesive_moments)

    def __update_unknowns_for_perturbation(self, e, perturbed_system_unknowns, 
                                           perturbed_solution_increments):
        """
        Update the relevant unknowns with perturbed values for numerical stiffness computation.

        Parameters:
            e: The element index.
            perturbed_system_unknowns: The perturbed system unknowns.
            perturbed_solution_increments: The perturbed solution increments.
        """
        # update the internal variables of the element
        self._update_element_internal_variables(
            e, perturbed_solution_increments)
        self.__update_element_position_jumps(e, perturbed_system_unknowns)
        # update the internal variables of the neighboring elements
        if (e == 0):  # for the left most element
            self._update_element_internal_variables(
                e+1, perturbed_solution_increments)
            self.__update_element_position_jumps(
                e+1, perturbed_system_unknowns)
        elif (e == self.function_space.E-1):  # for the right most element
            self._update_element_internal_variables(
                e-1, perturbed_solution_increments)
            self.__update_element_position_jumps(
                e-1, perturbed_system_unknowns)
        else:  # for the intermediate elements
            self._update_element_internal_variables(
                e-1, perturbed_solution_increments)
            self.__update_element_position_jumps(
                e-1, perturbed_system_unknowns)
            self._update_element_internal_variables(
                e+1, perturbed_solution_increments)
            self.__update_element_position_jumps(
                e+1, perturbed_system_unknowns)

    def __compute_element_numerical_stiffness(self, e, system_unknowns):
        """
        Compute the numerical stiffness matrix for an element based on the provided unknowns.

        Parameters:
            e: The element index.
            system_unknowns: The unknowns of the system.
        Returns:
            extended_element_dofs: The extended element dofs including the neighboring elements.
            element_numerical_stiffness: The computed numerical stiffness matrix for the element.
        """
        perturbation_factor = 1.0e-05
        np.random.seed(1234 + e)  # for reproducibility
        std_dev_system_unknowns = 0.01 * \
            np.maximum(np.abs(system_unknowns), 1.0e-03)
        perturbation_magnitudes = perturbation_factor * np.abs(
            np.random.normal(loc=system_unknowns, scale=std_dev_system_unknowns))
        # get the extended element dofs
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        global_element_dofs = self.function_space.global_connectivity[e:e+1].flatten()
        if (e == 0):  # for the left most element
            right_element_dofs_left_node = \
                self.function_space.global_connectivity[e+1:e+2].flatten()[0:dofs]
            extended_element_dofs = np.append(
                global_element_dofs, right_element_dofs_left_node)
        elif (e == self.function_space.E-1):  # for the right most element
            left_element_dofs_right_node = \
                self.function_space.global_connectivity[e-1:e].flatten()[dofs:dofspel]
            extended_element_dofs = np.append(
                left_element_dofs_right_node, global_element_dofs)
        else:  # for the intermediate elements
            left_element_dofs_right_node = \
                self.function_space.global_connectivity[e-1:e].flatten()[dofs:dofspel]
            right_element_dofs_left_node = \
                self.function_space.global_connectivity[e+1:e+2].flatten()[0:dofs]
            extended_element_dofs = np.append(
                left_element_dofs_right_node, np.append(
                    global_element_dofs, right_element_dofs_left_node))
        # perturb the element dofs individually to compute the numerical stiffness matrix
        element_numerical_stiffness = np.zeros(
            [extended_element_dofs.shape[0], extended_element_dofs.shape[0]])
        perturbed_system_unknowns = system_unknowns.copy()
        perturbed_solution_increments = np.zeros_like(system_unknowns)
        for i in range(0, extended_element_dofs.shape[0]):
            # reset the perturbed increments
            perturbed_solution_increments.fill(0.0)
            # perturb the i-th dof
            perturbed_solution_increments[extended_element_dofs[i]] = \
                perturbation_magnitudes[extended_element_dofs[i]]
            ####### positively perturb the unknowns #######
            perturbed_system_unknowns += \
                perturbed_solution_increments
            # update the internal variables for perturbation
            self.__update_unknowns_for_perturbation(
                e, perturbed_system_unknowns, perturbed_solution_increments)
            # compute the element internal forces for the positive perturbation
            element_internal_forces_positive_perturbation = self.compute_element_internal_forces(
                e, perturbed_system_unknowns[global_element_dofs], 
                self.orientation[e, :, :], self.curvature[e, :, :])
            ####### negatively perturb the unknowns #######
            perturbed_system_unknowns -= \
                2.0 * perturbed_solution_increments
            # update the internal variables for perturbation
            self.__update_unknowns_for_perturbation(
                e, perturbed_system_unknowns, -2.0 * perturbed_solution_increments)
            # compute the element internal forces for the negative perturbation
            element_internal_forces_negative_perturbation = self.compute_element_internal_forces(
                e, perturbed_system_unknowns[global_element_dofs], 
                self.orientation[e, :, :], self.curvature[e, :, :])
            # compute the element numerical stiffness matrix
            element_numerical_stiffness[:, i:i+1] += \
                (element_internal_forces_positive_perturbation - \
                 element_internal_forces_negative_perturbation) / \
                (2.0 * perturbation_magnitudes[extended_element_dofs[i]])
            ####### reset the unknowns and internal variables #######
            perturbed_system_unknowns += \
                perturbed_solution_increments
            self.__update_unknowns_for_perturbation(
                e, perturbed_system_unknowns, perturbed_solution_increments)
        return extended_element_dofs, element_numerical_stiffness

    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        """
        Compute the system stiffness matrix based on the provided unknowns and loads.

        Parameters:
            A: The stiffness matrix to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
            element_loads: The distributed loads on the elements.
        """
        # update the position jumps at the element boundaries before computing the residual
        # NOTE: This is necessary to ensure that the stiffness is computed with the correct
        # position jumps.
        self.__update_system_position_jumps(system_unknowns)
        for i in range(0, self.function_space.E):
            extended_element_dofs, element_numerical_stiffness = \
                self.__compute_element_numerical_stiffness(i, system_unknowns)
            A[np.ix_(extended_element_dofs, extended_element_dofs)] += \
                element_numerical_stiffness
        # assemble the interface stiffness terms
        N_left_interface, _ = self.function_space.compute_shapes(1.0)
        N_right_interface, _ = self.function_space.compute_shapes(-1.0)
        # loop over the interface elements
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            ############## penalty term tangents ##############
            # left-left terms
            A[np.ix_(global_element_dofs_left[self.function_space.local_translational_dofs],
                     global_element_dofs_left[self.function_space.local_translational_dofs])] += \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaP*((self.material.E*self.material.A)/self.function_space.elL) *
                     np.matmul(np.transpose(N_left_interface), N_left_interface))
            A[np.ix_(global_element_dofs_left[self.function_space.local_rotational_dofs],
                     global_element_dofs_left[self.function_space.local_rotational_dofs])] += \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaT*((self.material.E*self.material.I)/self.function_space.elL) *
                     np.matmul(np.transpose(N_left_interface), N_left_interface))
            # left-right terms
            A[np.ix_(global_element_dofs_left[self.function_space.local_translational_dofs],
                     global_element_dofs_right[self.function_space.local_translational_dofs])] -= \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaP*((self.material.E*self.material.A)/self.function_space.elL) *
                     np.matmul(np.transpose(N_left_interface), N_right_interface))
            A[np.ix_(global_element_dofs_left[self.function_space.local_rotational_dofs],
                     global_element_dofs_right[self.function_space.local_rotational_dofs])] -= \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaT*((self.material.E*self.material.I)/self.function_space.elL) *
                     np.matmul(np.transpose(N_left_interface), N_right_interface))
            # right-left terms
            A[np.ix_(global_element_dofs_right[self.function_space.local_translational_dofs],
                     global_element_dofs_left[self.function_space.local_translational_dofs])] -= \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaP*((self.material.E*self.material.A)/self.function_space.elL) *
                     np.matmul(np.transpose(N_right_interface), N_left_interface))
            A[np.ix_(global_element_dofs_right[self.function_space.local_rotational_dofs],
                     global_element_dofs_left[self.function_space.local_rotational_dofs])] -= \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaT*((self.material.E*self.material.I)/self.function_space.elL) *
                     np.matmul(np.transpose(N_right_interface), N_left_interface))
            # right-right terms
            A[np.ix_(global_element_dofs_right[self.function_space.local_translational_dofs],
                     global_element_dofs_right[self.function_space.local_translational_dofs])] += \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaP*((self.material.E*self.material.A)/self.function_space.elL) *
                     np.matmul(np.transpose(N_right_interface), N_right_interface))
            A[np.ix_(global_element_dofs_right[self.function_space.local_rotational_dofs],
                     global_element_dofs_right[self.function_space.local_rotational_dofs])] += \
                (1.0 - self.internal_variables[i:i+1, 1:2]) * \
                    (self.betaT*((self.material.E*self.material.I)/self.function_space.elL) *
                     np.matmul(np.transpose(N_right_interface), N_right_interface))
