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
        # containers to store the orientation and curvature at the nodes
        # NOTE: The curvature at the nodes is not used in the weak form, but it is needed for
        # post-processing (to compute the internal moments).
        self.orientation_nodes = np.zeros(
            [self.function_space.N, self.function_space.dim])
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

    def _update_element_internal_variables(self, e, element_unknowns_increment):
        """
        Update the internal variables of an element based on the current system unknowns increment.

        This method updates the orientation and curvature of the element at each quadrature point 
        based on the current system unknowns increment.

        Parameters:
            e: The index of the element.
            element_unknowns_increment: The increment in the element unknowns.
        """
        N = self.function_space.shape_functions
        element_rotation_increment = element_unknowns_increment[
            self.function_space.local_rotational_dofs]
        # compute the "additive" rotation increment and its derivative
        dtheta = np.matmul(N, element_rotation_increment)[..., 0]
        dtheta_prime = self._compute_element_dof_derivatives(
            e, element_unknowns_increment, self.function_space.local_rotational_dofs)[..., 0]
        # compute the transformation and R matrix
        transformation_matrix = self._compute_transformation_matrix(
            self.orientation[e, :, :])
        transformation_matrix_inv = self._compute_transformation_matrix_inverse(
            self.orientation[e, :, :])
        element_orientation_derivative = np.matmul(
            transformation_matrix_inv, self.curvature[e, :, :][..., None])[..., 0]
        R_matrix = self._compute_R_matrix(
            self.orientation[e, :, :], element_orientation_derivative)
        # compute the rotation increment multiplier matrix
        curvature_skew_matrix = skew_symmetric_matrices(
            self.curvature[e, :, :])
        rotation_increment_multiplier_matrix = \
            R_matrix + np.matmul(curvature_skew_matrix, transformation_matrix)
        # transform the additive rotation increments to multiplicative increments
        dtheta_prime = np.matmul(
            transformation_matrix, dtheta_prime[..., None])[..., 0] + \
            np.matmul(rotation_increment_multiplier_matrix, dtheta[..., None])[..., 0]
        dtheta = np.matmul(
            transformation_matrix, dtheta[..., None])[..., 0]
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
        # transform the additive updates of rotation increments to multiplicative updates
        transformation_matrices = \
            self._compute_transformation_matrix(rotation_solution_vectors)
        rotation_increment_vectors = \
            np.matmul(transformation_matrices, rotation_increment_vectors[..., None])[..., 0]
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
                i, system_unknowns_increment[global_element_dofs])
        ########## update the curvature at the nodes ##########
        # NOTE: Here we assume that there are only two nodes per element!!!
        N_left_node, _ = self.function_space.compute_shapes(-1.0)
        N_right_node, _ = self.function_space.compute_shapes(1.0)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_rotation_increment = system_unknowns_increment[
                global_element_dofs][self.function_space.local_rotational_dofs]
            # compute the "additive" rotation increment and its derivative
            dtheta_left_node = np.matmul(
                N_left_node, element_rotation_increment)[..., 0]
            dtheta_right_node = np.matmul(
                N_right_node, element_rotation_increment)[..., 0]
            dtheta_prime_node = self._compute_element_dof_derivatives(
                i, system_unknowns_increment[global_element_dofs],
                self.function_space.local_rotational_dofs, location="Nodes")[..., 0]
            dtheta_prime_left_node = dtheta_prime_node[0, :]
            dtheta_prime_right_node = dtheta_prime_node[1, :]
            # compute the transformation and R matrices
            transformation_matrix_left_node = self._compute_transformation_matrix(
                self.orientation_nodes[i:i+1, :])
            transformation_matrix_right_node = self._compute_transformation_matrix(
                self.orientation_nodes[i+1:i+2, :])
            transformation_matrix_left_node_inv = self._compute_transformation_matrix_inverse(
                self.orientation_nodes[i:i+1, :])
            transformation_matrix_right_node_inv = self._compute_transformation_matrix_inverse(
                self.orientation_nodes[i+1:i+2, :])
            element_orientation_derivative_left_node = np.matmul(
                transformation_matrix_left_node_inv, self.curvature_nodes[i:i+1, :][..., None])[..., 0]
            element_orientation_derivative_right_node = np.matmul(
                transformation_matrix_right_node_inv, self.curvature_nodes[i+1:i+2, :][..., None])[..., 0]
            R_matrix_left_node = self._compute_R_matrix(
                self.orientation_nodes[i:i+1, :], element_orientation_derivative_left_node)
            R_matrix_right_node = self._compute_R_matrix(
                self.orientation_nodes[i+1:i+2, :], element_orientation_derivative_right_node)
            # compute the rotation increment multiplier matrix
            curvature_skew_matrix_left_node = skew_symmetric_matrices(
                self.curvature_nodes[i:i+1, :])
            curvature_skew_matrix_right_node = skew_symmetric_matrices(
                self.curvature_nodes[i+1:i+2, :])
            rotation_increment_multiplier_matrix_left_node = \
                R_matrix_left_node + np.matmul(curvature_skew_matrix_left_node, transformation_matrix_left_node)
            rotation_increment_multiplier_matrix_right_node = \
                R_matrix_right_node + np.matmul(curvature_skew_matrix_right_node, transformation_matrix_right_node)
            # transform the additive rotation increments to multiplicative increments
            dtheta_prime_left_node = np.matmul(
                transformation_matrix_left_node, dtheta_prime_left_node[..., None])[..., 0] + \
                np.matmul(rotation_increment_multiplier_matrix_left_node, dtheta_left_node[..., None])[..., 0]
            dtheta_prime_right_node = np.matmul(
                transformation_matrix_right_node, dtheta_prime_right_node[..., None])[..., 0] + \
                np.matmul(rotation_increment_multiplier_matrix_right_node, dtheta_right_node[..., None])[..., 0]
            dtheta_left_node = np.matmul(
                transformation_matrix_left_node, dtheta_left_node[..., None])[..., 0]
            dtheta_right_node = np.matmul(
                transformation_matrix_right_node, dtheta_right_node[..., None])[..., 0]            
            # incremental transformation matrices
            incremental_transformation_matrix_left_node = self._compute_transformation_matrix(
                dtheta_left_node)[0, ...]
            incremental_transformation_matrix_right_node = self._compute_transformation_matrix(
                dtheta_right_node)[0, ...]
            incremental_rotation_tensor_left_node = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta_left_node))
            incremental_rotation_tensor_right_node = \
                quaternion.as_rotation_matrix(
                    quaternion.from_rotation_vector(dtheta_right_node))
            if (self.function_space.discretization_type == "CG"):  # for CG discretization
                if (i == 0):  # only for the first element
                    # update the orientation at the left node
                    self.orientation_nodes[0, :] = \
                        quaternion.as_rotation_vector(
                            quaternion.from_rotation_vector(dtheta_left_node) * \
                                quaternion.from_rotation_vector(self.orientation_nodes[0, :]))
                    # update the curvature at the left node
                    self.curvature_nodes[0, :] = np.matmul(
                        incremental_transformation_matrix_left_node, dtheta_prime_left_node[..., None])[..., 0] + \
                        np.matmul(incremental_rotation_tensor_left_node, 
                                  self.curvature_nodes[0, :][..., None])[..., 0]
                # update the orientation at the right node
                self.orientation_nodes[i+1, :] = \
                    quaternion.as_rotation_vector(
                        quaternion.from_rotation_vector(dtheta_right_node) * \
                            quaternion.from_rotation_vector(self.orientation_nodes[i+1, :]))
                # update the curvature at the right node
                self.curvature_nodes[i+1, :] = np.matmul(
                    incremental_transformation_matrix_right_node, dtheta_prime_right_node[..., None])[..., 0] + \
                    np.matmul(incremental_rotation_tensor_right_node, 
                              self.curvature_nodes[i+1, :][..., None])[..., 0]
            elif (self.function_space.discretization_type == "DG"):  # for DG discretization
                # update the orientation at the left node
                self.orientation_nodes[2*i, :] = \
                    quaternion.as_rotation_vector(
                        quaternion.from_rotation_vector(dtheta_left_node) * \
                            quaternion.from_rotation_vector(self.orientation_nodes[2*i, :]))
                # update the curvature at the left node
                self.curvature_nodes[2*i, :] = np.matmul(
                    incremental_transformation_matrix_left_node, dtheta_prime_left_node[..., None])[..., 0] + \
                    np.matmul(incremental_rotation_tensor_left_node,
                              self.curvature_nodes[2*i, :][..., None])[..., 0]
                # update the orientation at the right node
                self.orientation_nodes[2*i+1, :] = \
                    quaternion.as_rotation_vector(
                        quaternion.from_rotation_vector(dtheta_right_node) * \
                            quaternion.from_rotation_vector(self.orientation_nodes[2*i+1, :]))
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
        # compute the transformation matrix
        transformation_matrix = self._compute_transformation_matrix(
            element_orientations)
        # compute the derivative of the orientation
        transformation_matrix_inv = self._compute_transformation_matrix_inverse(
            element_orientations)
        element_orientations_derivative = np.matmul(
            transformation_matrix_inv, element_curvatures[..., None])[..., 0]
        # compute the moment multiplier matrix
        R_matrix = self._compute_R_matrix(
            element_orientations, element_orientations_derivative)
        curvature_skew_matrix = skew_symmetric_matrices(
            element_curvatures)
        moment_multiplier_matrix = R_matrix + \
            np.matmul(curvature_skew_matrix, transformation_matrix)
        # assemble the internal forces
        internal_forces_integrand = np.matmul(
            np.transpose(Np, axes=(0, 2, 1)), internal_forces)
        element_internal_forces[self.function_space.local_translational_dofs] += \
            np.sum(internal_forces_integrand *
                   self.function_space.JxW, axis=0, keepdims=False)
        # assemble the internal moments
        internal_moments_integrand = np.matmul(
            np.transpose(Np, axes=(0, 2, 1)), np.matmul(
                np.transpose(transformation_matrix, axes=(0, 2, 1)), internal_moments))
        internal_moments_integrand += np.matmul(
            np.transpose(N, axes=(0, 2, 1)), np.matmul(
                np.transpose(moment_multiplier_matrix, axes=(0, 2, 1)), internal_moments))
        internal_moments_integrand -= np.matmul(
            np.transpose(N, axes=(0, 2, 1)), np.matmul(
                np.transpose(transformation_matrix, axes=(0, 2, 1)), 
                    np.cross(rp, internal_forces, axis=1)))
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
        global_element_dofs = self.function_space.global_connectivity[e:e+1].flatten()
        std_dev_element_unknowns = 0.01 * \
            np.maximum(np.abs(system_unknowns[global_element_dofs]), 1.0e-03)
        perturbation_magnitudes = perturbation_factor * np.abs(
            np.random.normal(loc=system_unknowns[global_element_dofs], 
                             scale=std_dev_element_unknowns))
        # perturb the element dofs individually to compute the numerical stiffness matrix
        element_numerical_stiffness = np.zeros(
            [global_element_dofs.shape[0], global_element_dofs.shape[0]])
        perturbed_element_unknowns = system_unknowns[global_element_dofs].copy()
        perturbed_solution_increments = np.zeros_like(system_unknowns[global_element_dofs])
        for i in range(0, global_element_dofs.shape[0]):
            # reset the perturbed increments
            perturbed_solution_increments.fill(0.0)
            # perturb the i-th dof
            perturbed_solution_increments[i] = perturbation_magnitudes[i]
            ####### positively perturb the unknowns #######
            perturbed_element_unknowns += perturbed_solution_increments
            # update the internal variables of the element
            self._update_element_internal_variables(
                e, perturbed_solution_increments)
            # compute the element internal forces for the positive perturbation
            element_internal_forces_positive_perturbation = self.compute_element_internal_forces(
                e, perturbed_element_unknowns, self.orientation[e, :, :], self.curvature[e, :, :])
            ####### negatively perturb the unknowns #######
            perturbed_element_unknowns -= 2.0 * perturbed_solution_increments
            # update the internal variables of the element
            self._update_element_internal_variables(
                e, -2.0 * perturbed_solution_increments)
            # compute the element internal forces for the negative perturbation
            element_internal_forces_negative_perturbation = self.compute_element_internal_forces(
                e, perturbed_element_unknowns, self.orientation[e, :, :], self.curvature[e, :, :])
            # compute the element numerical stiffness matrix
            element_numerical_stiffness[:, i:i+1] += \
                (element_internal_forces_positive_perturbation - \
                 element_internal_forces_negative_perturbation) / \
                (2.0 * perturbation_magnitudes[i])
            ####### reset the unknowns and internal variables #######
            perturbed_element_unknowns += \
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
                A[np.ix_(global_element_dofs, global_element_dofs)] += \
                    self.__compute_element_numerical_stiffness(i, system_unknowns)
        # add the contribution of the nodal moments to the stiffness matrix
        dofs = self.function_space.dof
        for i in range(0, self.function_space.N):
            nodal_orientations = system_unknowns[(dofs*i)+3:(dofs*i)+6, :]
            nodal_moments = nodal_loads[(dofs*i)+3:(dofs*i)+6, :]
            # perturb the nodal orientations to compute the numerical stiffness matrix
            perturbation_factor = 1.0e-05
            np.random.seed(1234 + i)  # for reproducibility
            std_dev_nodal_orientations = 0.01 * \
                np.maximum(np.abs(nodal_orientations), 1.0e-03)
            perturbation_magnitudes = perturbation_factor * np.abs(
                np.random.normal(loc=nodal_orientations, 
                                 scale=std_dev_nodal_orientations))
            # perturb the nodal orientations individually
            nodal_moments_stiffness_contribution = np.zeros([3, 3])
            perturbed_nodal_orientations = nodal_orientations.copy()
            for j in range(0, 3):
                # perturb the j-th orientation positively
                perturbed_nodal_orientations[j] += perturbation_magnitudes[j]
                perturbed_transformation_matrix = self._compute_transformation_matrix(
                    perturbed_nodal_orientations.T)[0, ...]
                nodal_moments_residual_positive_perturbation = \
                    np.matmul(np.transpose(perturbed_transformation_matrix), nodal_moments)
                # perturb the j-th orientation negatively
                perturbed_nodal_orientations[j] -= 2.0 * perturbation_magnitudes[j]
                perturbed_transformation_matrix = self._compute_transformation_matrix(
                    perturbed_nodal_orientations.T)[0, ...]
                nodal_moments_residual_negative_perturbation = \
                    np.matmul(np.transpose(perturbed_transformation_matrix), nodal_moments)
                # compute the nodal moments contribution
                nodal_moments_stiffness_contribution[:, j:j+1] += \
                    (nodal_moments_residual_positive_perturbation -
                     nodal_moments_residual_negative_perturbation) / \
                    (2.0 * perturbation_magnitudes[j])
                # reset the perturbed nodal orientations
                perturbed_nodal_orientations[j] += perturbation_magnitudes[j]
            # add the nodal moments stiffness contribution to the stiffness matrix
            A[(dofs*i)+3:(dofs*i)+6, (dofs*i)+3:(dofs*i)+6] -= \
                nodal_moments_stiffness_contribution

    def add_nodal_loads_to_residual(self, f, system_unknowns, nodal_loads):
        """
        Add the nodal loads to the residual vector.

        Parameters:
            f: The residual vector to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
        """
        dofs = self.function_space.dof
        updated_nodal_loads = np.zeros_like(nodal_loads)
        for i in range(0, self.function_space.N):
            # add the nodal forces
            updated_nodal_loads[dofs*i:(dofs*i)+3,
                                :] += nodal_loads[dofs*i:(dofs*i)+3, :]
            # transform the nodal moments and add them to the residual
            nodal_orientations = system_unknowns[(dofs*i)+3:(dofs*i)+6, :]
            nodal_transformation_matrix = self._compute_transformation_matrix(
                nodal_orientations.T)[0, ...]
            updated_nodal_loads[(dofs*i)+3:(dofs*i)+6, :] += \
                np.matmul(np.transpose(nodal_transformation_matrix),
                          nodal_loads[(dofs*i)+3:(dofs*i)+6, :])
        f += updated_nodal_loads

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
            internal_forces, internal_moments = self._compute_internal_forces_and_moments(
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
        # NOTE: We are using the self.curvature_nodes container to store the curvature at the 
        # boundaries of the elements since we have two nodes per element which are the left and right 
        # nodes. So instead of creating a new container, we are reusing the curvature_nodes container 
        # to store the curvature at the boundaries of the elements which is needed in the DG weak 
        # form to compute the moments at the element interfaces. This will not work if we have more
        # than two nodes per element i.e. for higher order elements.

    def compute_interface_residual(self, e, element_unknowns_left, element_unknowns_right):
        """
        Compute the residual at an interface based on the provided unknowns.

        Parameters:
            e: The (interface) element index.
            element_unknowns_left: The unknowns of the left element.
            element_unknowns_right: The unknowns of the right element.
        Returns:
            interface_residual: The computed interface residual.
        """
        dofspel = self.function_space.dof*self.function_space.npel
        interface_residual = np.zeros([2*dofspel, 1])
        # shape functions at the interface
        N_left_interface, _ = self.function_space.compute_shapes(1.0)
        N_right_interface, _ = self.function_space.compute_shapes(-1.0)
        # nodal orientations for left and right elements
        nodal_orientations_left = \
            element_unknowns_left[self.function_space.local_rotational_dofs].reshape(
                -1, self.function_space.dim)
        nodal_orientations_right = \
            element_unknowns_right[self.function_space.local_rotational_dofs].reshape(
                -1, self.function_space.dim)
        # nodal curvatures for left and right elements
        nodal_curvatures_left = np.stack(
            [self.curvature_nodes[2*e, :], self.curvature_nodes[2*e+1, :]], axis=0)
        nodal_curvatures_right = np.stack(
            [self.curvature_nodes[2*e+2, :], self.curvature_nodes[2*e+3, :]], axis=0)
        # compute the internal forces and moments at the nodes for the left and right elements
        internal_forces_left_element, internal_moments_left_element = \
            self._compute_internal_forces_and_moments(e, element_unknowns_left,
                                                      nodal_orientations_left,
                                                      nodal_curvatures_left,
                                                      location="Nodes")
        internal_forces_right_element, internal_moments_right_element = \
            self._compute_internal_forces_and_moments(e+1, element_unknowns_right,
                                                      nodal_orientations_right,
                                                      nodal_curvatures_right,
                                                      location="Nodes")
        # compute the transformation matrices at the interface
        transformation_matrix_left_interface = self._compute_transformation_matrix(
            nodal_orientations_left[1:2, :])[0, ...]
        transformation_matrix_right_interface = self._compute_transformation_matrix(
            nodal_orientations_right[0:1, :])[0, ...]
        average_internal_forces_interface = 0.50 * \
            (internal_forces_left_element[1, ...] +
             internal_forces_right_element[0, ...])
        average_internal_moments_interface = 0.50 * \
            (np.matmul(np.transpose(transformation_matrix_left_interface),
                       internal_moments_left_element[1, ...]) +
             np.matmul(np.transpose(transformation_matrix_right_interface),
                       internal_moments_right_element[0, ...]))
        # compute the dof jumps at the interface
        r_jump_interface = \
            np.matmul(N_right_interface, 
                      element_unknowns_right[self.function_space.local_translational_dofs]) - \
            np.matmul(N_left_interface, 
                      element_unknowns_left[self.function_space.local_translational_dofs])
        # NOTE: Although we attempt to interpolate the rotational dofs at the interface here, in reality
        # only their incremental vectors are interpolated. We do not interpolate the total rotational
        # vectors directly as that would result in a loss of objectivity. We can still consider this 
        # to be a good approximation since the difference in the rotational dofs at the interface will 
        # be small at end of each load/time step i.e. for a converged configuration.
        psi_jump_interface = \
            np.matmul(N_right_interface, 
                      element_unknowns_right[self.function_space.local_rotational_dofs]) - \
            np.matmul(N_left_interface, 
                      element_unknowns_left[self.function_space.local_rotational_dofs])
        # assemble the interface residual terms
        ############# flux terms #############
        interface_residual[self.function_space.local_translational_dofs] -= \
            np.matmul(np.transpose(N_left_interface), 
                      average_internal_forces_interface)
        interface_residual[self.function_space.local_rotational_dofs] -= \
            np.matmul(np.transpose(N_left_interface), 
                      average_internal_moments_interface)
        interface_residual[dofspel + self.function_space.local_translational_dofs] += \
            np.matmul(np.transpose(N_right_interface), average_internal_forces_interface)
        interface_residual[dofspel + self.function_space.local_rotational_dofs] += \
            np.matmul(np.transpose(N_right_interface), average_internal_moments_interface)
        ############# penalty terms #############
        C_F = np.zeros([self.function_space.dim, self.function_space.dim])
        C_F[0, 0] = self.material.E * self.material.A
        C_F[1, 1] = self.material.G * self.material.A_red
        C_F[2, 2] = self.material.G * self.material.A_red
        C_M = np.zeros([self.function_space.dim, self.function_space.dim])
        C_M[0, 0] = self.material.G * self.material.I_T
        C_M[1, 1] = self.material.E * self.material.I
        C_M[2, 2] = self.material.E * self.material.I_minor
        # compute the transformed constitutive matrices
        orientations_tensor_left_interface = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(nodal_orientations_left[1, :]))
        orientations_tensor_right_interface = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(nodal_orientations_right[0, :]))
        C_F_transformed_left_interface = np.matmul(
            orientations_tensor_left_interface, np.matmul(
                C_F, np.transpose(orientations_tensor_left_interface)))
        C_F_transformed_right_interface = np.matmul(
            orientations_tensor_right_interface, np.matmul(
                C_F, np.transpose(orientations_tensor_right_interface)))
        C_F_transformed_average_interface = 0.50 * \
            (C_F_transformed_left_interface + C_F_transformed_right_interface)
        C_M_transformed_left_interface = np.matmul(
            orientations_tensor_left_interface, np.matmul(
                C_M, np.transpose(orientations_tensor_left_interface)))
        C_M_transformed_right_interface = np.matmul(
            orientations_tensor_right_interface, np.matmul(
                C_M, np.transpose(orientations_tensor_right_interface)))
        C_M_transformed_left_interface = np.matmul(np.transpose(
            transformation_matrix_left_interface), np.matmul(
                C_M_transformed_left_interface, transformation_matrix_left_interface))
        C_M_transformed_right_interface = np.matmul(np.transpose(
            transformation_matrix_right_interface), np.matmul(
                C_M_transformed_right_interface, transformation_matrix_right_interface))
        C_M_transformed_average_interface = 0.50 * \
            (C_M_transformed_left_interface + C_M_transformed_right_interface)
        # compute the penalty forces and moments
        penalty_forces = (self.betaP / self.function_space.elL) * \
            np.matmul(C_F_transformed_average_interface, r_jump_interface)
        penalty_moments = (self.betaT / self.function_space.elL) * \
            np.matmul(C_M_transformed_average_interface, psi_jump_interface)
        interface_residual[self.function_space.local_translational_dofs] -= \
            np.matmul(np.transpose(N_left_interface), penalty_forces)
        interface_residual[self.function_space.local_rotational_dofs] -= \
            np.matmul(np.transpose(N_left_interface), penalty_moments)
        interface_residual[dofspel + self.function_space.local_translational_dofs] += \
            np.matmul(np.transpose(N_right_interface), penalty_forces)
        interface_residual[dofspel + self.function_space.local_rotational_dofs] += \
            np.matmul(np.transpose(N_right_interface), penalty_moments)
        return interface_residual

    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        """
        Compute the system residual based on the provided unknowns and element loads.

        Parameters:
            f: The force vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
            update_internal: A boolean indicating whether to update the internal variables.
        """
        # assemble the bulk terms using the method in the parent class
        super().compute_system_residual(f, system_unknowns, element_loads, update_internal)
        # assemble the interface terms
        for i in range(0, self.function_space.E-1):  # loop over the interface elements
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # unknowns of the left and right elements
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            # compute the interface residual and assemble it to the global residual
            interface_residual = self.compute_interface_residual(
                i, element_unknowns_left, element_unknowns_right)
            global_element_dofs_interface = np.concatenate(
                [global_element_dofs_left, global_element_dofs_right])
            f[global_element_dofs_interface] -= interface_residual

    def __compute_numerical_interface_stiffness(self, e, element_unknowns_left,
                                                element_unknowns_right):
        """
        Compute the numerical stiffness matrix at an interface based on the provided unknowns.

        Parameters:
            e: The (interface) element index.
            element_unknowns_left: The unknowns of the left element.
            element_unknowns_right: The unknowns of the right element.
        Returns:
            interface_numerical_stiffness: The computed numerical stiffness matrix at the interface.
        """
        dofspel = self.function_space.dof*self.function_space.npel
        interface_numerical_stiffness = np.zeros([2*dofspel, 2*dofspel])
        perturbation_factor = 1.0e-05
        np.random.seed(1234 + e)  # for reproducibility
        std_dev_element_unknowns_left = 0.01 * \
            np.maximum(np.abs(element_unknowns_left), 1.0e-03)
        std_dev_element_unknowns_right = 0.01 * \
            np.maximum(np.abs(element_unknowns_right), 1.0e-03)
        perturbation_magnitudes_left = perturbation_factor * np.abs(
            np.random.normal(loc=element_unknowns_left, 
                             scale=std_dev_element_unknowns_left))
        perturbation_magnitudes_right = perturbation_factor * np.abs(
            np.random.normal(loc=element_unknowns_right, 
                             scale=std_dev_element_unknowns_right))
        # perturb the left element dofs individually to compute the numerical stiffness matrix
        perturbed_element_unknowns = element_unknowns_left.copy()
        perturbed_solution_increments = np.zeros_like(element_unknowns_left)
        for i in range(0, dofspel):
            # reset the perturbed increments
            perturbed_solution_increments.fill(0.0)
            # perturb the i-th dof of the left element
            perturbed_solution_increments[i] = perturbation_magnitudes_left[i]
            ####### positively perturb the unknowns #######
            perturbed_element_unknowns += perturbed_solution_increments
            # update the internal variables of the left element
            self._update_element_internal_variables(
                e, perturbed_solution_increments)
            # compute the interface residual for the positive perturbation
            interface_residual_positive_perturbation_left = self.compute_interface_residual(
                e, perturbed_element_unknowns, element_unknowns_right)
            ####### negatively perturb the unknowns #######
            perturbed_element_unknowns -= 2.0 * perturbed_solution_increments
            # update the internal variables of the left element
            self._update_element_internal_variables(
                e, -2.0 * perturbed_solution_increments)
            # compute the interface residual for the negative perturbation
            interface_residual_negative_perturbation_left = self.compute_interface_residual(
                e, perturbed_element_unknowns, element_unknowns_right)
            # compute the interface numerical stiffness matrix for the left element
            interface_numerical_stiffness[:, i:i+1] += \
                (interface_residual_positive_perturbation_left -
                 interface_residual_negative_perturbation_left) / \
                (2.0 * perturbation_magnitudes_left[i])
            ####### reset the unknowns and internal variables #######
            perturbed_element_unknowns += \
                perturbed_solution_increments
            self._update_element_internal_variables(
                e, perturbed_solution_increments)
        # perturb the right element dofs individually to compute the numerical stiffness matrix
        perturbed_element_unknowns = element_unknowns_right.copy()
        for i in range(dofspel, 2*dofspel):
            # reset the perturbed increments
            perturbed_solution_increments.fill(0.0)
            # perturb the i-th dof of the right element
            perturbed_solution_increments[i - dofspel] = perturbation_magnitudes_right[i - dofspel]
            ####### positively perturb the unknowns #######
            perturbed_element_unknowns += perturbed_solution_increments
            # update the internal variables of the right element
            self._update_element_internal_variables(
                e+1, perturbed_solution_increments)
            # compute the interface residual for the positive perturbation
            interface_residual_positive_perturbation_right = self.compute_interface_residual(
                e, element_unknowns_left, perturbed_element_unknowns)
            ####### negatively perturb the unknowns #######
            perturbed_element_unknowns -= 2.0 * perturbed_solution_increments
            # update the internal variables of the right element
            self._update_element_internal_variables(
                e+1, -2.0 * perturbed_solution_increments)
            # compute the interface residual for the negative perturbation
            interface_residual_negative_perturbation_right = self.compute_interface_residual(
                e, element_unknowns_left, perturbed_element_unknowns)
            # compute the interface numerical stiffness matrix for the right element
            interface_numerical_stiffness[:, i:i+1] += \
                (interface_residual_positive_perturbation_right -
                 interface_residual_negative_perturbation_right) / \
                (2.0 * perturbation_magnitudes_right[i - dofspel])
            ####### reset the unknowns and internal variables #######
            perturbed_element_unknowns += \
                perturbed_solution_increments
            self._update_element_internal_variables(
                e+1, perturbed_solution_increments)
        return interface_numerical_stiffness

    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        """
        Compute the system stiffness matrix based on the provided unknowns and loads.

        Parameters:
            A: The stiffness matrix to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
            element_loads: The distributed loads on the elements.
        """
        # assemble the bulk terms using the method in the parent class
        super().compute_system_stiffness(A, system_unknowns, nodal_loads, element_loads)
        # assemble the interface stiffness terms
        for i in range(0, self.function_space.E-1):  # loop over the interface elements
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # unknowns of the left and right elements
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            # compute the interface numerical stiffness matrix and assemble it to the global stiffness matrix
            interface_numerical_stiffness = self.__compute_numerical_interface_stiffness(
                i, element_unknowns_left, element_unknowns_right)
            global_element_dofs_interface = np.concatenate(
                [global_element_dofs_left, global_element_dofs_right])
            A[np.ix_(global_element_dofs_interface, global_element_dofs_interface)] += \
                interface_numerical_stiffness
