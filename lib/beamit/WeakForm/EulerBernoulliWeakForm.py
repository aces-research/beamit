import numpy as np
from beamit.WeakForm.WeakForm import WeakForm


class EulerBernoulliWeakFormCG(WeakForm):
    
    def __init__(self, function_space, material):
        """
        Initialize the EulerBernoulliWeakFormCG class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
        """
        # initialize the parent (WeakForm) class
        WeakForm.__init__(self, function_space, material)

    # Function to compute element internal forces
    def compute_element_internal_forces(self, element_unknowns):
        """
        Compute the internal forces for an element based on the provided unknowns.

        Parameters:
            element_unknowns: The unknowns of the element.
        Returns:
            f_int: The computed internal forces for the element.
        """
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        ux = np.matmul(phix, element_unknowns)
        wxx = np.matmul(Nxx, element_unknowns)
        integrand = self.material.E*self.material.A*np.matmul(np.transpose(phix, axes=(0, 2, 1)), ux) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(Nxx, axes=(0, 2, 1)), wxx)
        return np.sum(integrand*self.function_space.JxW, axis=0, keepdims=False)

    # Function to compute element internal stiffness
    def compute_element_internal_stiffness(self, element_unknowns):
        """
        Compute the internal stiffness for an element based on the provided unknowns.

        Parameters:
            element_unknowns: The unknowns of the element.
        Returns:
            K_int: The computed internal stiffness for the element.
        """
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        integrand = self.material.E*self.material.A*np.matmul(np.transpose(phix, axes=(0, 2, 1)), phix) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(Nxx, axes=(0, 2, 1)), Nxx)
        return np.sum(integrand*self.function_space.JxW, axis=0, keepdims=False)

    # Function to compute the system mass
    def compute_system_mass(self, M, lump=True, **kwargs):
        """
        Compute the system mass matrix.

        Parameters:
            M: The mass matrix to be assembled.
            lump: If True, apply lumping to the mass matrix.
            **kwargs: Optional keyword arguments.
        """
        phit = np.transpose(self.function_space.lagrange_shape_functions, axes=(0, 2, 1))
        Nt = np.transpose(self.function_space.hermite_shape_functions, axes=(0, 2, 1))
        axial_integrand = self.material.rho*self.material.A * \
                                np.matmul(phit, self.function_space.lagrange_shape_functions)
        bending_integrand = self.material.rho*self.material.A * \
                                np.matmul(Nt, self.function_space.hermite_shape_functions)
        
        M_el_axial = np.sum(axial_integrand*self.function_space.JxW, axis=0, keepdims=False)
        M_el_bending = np.sum(bending_integrand*self.function_space.JxW, axis=0, keepdims=False)
        if (lump):
            # apply row sum to the axial mass matrix
            diag_elements = np.sum(M_el_axial, axis=1)
            M_el_axial = np.zeros([self.function_space.npel*self.function_space.dof, \
                                            self.function_space.npel*self.function_space.dof])
            np.fill_diagonal(M_el_axial, diag_elements)
            # apply "special lumping" to the bending mass matrix
            sum_all_entries = np.sum(M_el_bending)
            sum_diag_entries = np.sum(np.diag(M_el_bending))
            diag_elements = (sum_all_entries / sum_diag_entries) * np.diag(M_el_bending)
            M_el_bending = np.zeros([self.function_space.npel*self.function_space.dof, \
                                            self.function_space.npel*self.function_space.dof])
            np.fill_diagonal(M_el_bending, diag_elements)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            M[np.ix_(global_element_dofs, global_element_dofs)] += (M_el_axial + M_el_bending)

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
        _, phixi_left_node = self.function_space.compute_lagrange_shapes(-1.0)
        phix_left_node = phixi_left_node*(1.0/self.function_space.jacobian)
        _, _, Nxixi_left_node, Nxixixi_left_node = self.function_space.compute_hermite_shapes(-1.0)
        Nxx_left_node = Nxixi_left_node*((1.0/self.function_space.jacobian)**2.0)
        Nxxx_left_node = Nxixixi_left_node*((1.0/self.function_space.jacobian)**3.0)
        _, phixi_right_node = self.function_space.compute_lagrange_shapes(1.0)
        phix_right_node = phixi_right_node*(1.0/self.function_space.jacobian)
        _, _, Nxixi_right_node, Nxixixi_right_node = self.function_space.compute_hermite_shapes(1.0)
        Nxx_right_node = Nxixi_right_node*((1.0/self.function_space.jacobian)**2.0)
        Nxxx_right_node = Nxixixi_right_node*((1.0/self.function_space.jacobian)**3.0)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            ux_right_node = np.matmul(phix_right_node, element_unknowns)
            wxx_right_node = np.matmul(Nxx_right_node, element_unknowns)
            wxxx_right_node = np.matmul(Nxxx_right_node, element_unknowns)
            axial_force_right_node = self.material.E*self.material.A*ux_right_node
            shear_force_right_node = -self.material.E*self.material.I*wxxx_right_node
            bending_moment_right_node = -self.material.E*self.material.I*wxx_right_node
            f[global_element_dofs_right_node[0:1]] += axial_force_right_node
            f[global_element_dofs_right_node[1:2]] += shear_force_right_node
            f[global_element_dofs_right_node[2:3]] += bending_moment_right_node
            # Assuming the elements are connected like a simple chain!!!
            if (i == 0): # only for the first element
                global_element_dofs_left_node = global_element_dofs[0:dofs]
                ux_left_node = np.matmul(phix_left_node, element_unknowns)
                wxx_left_node = np.matmul(Nxx_left_node, element_unknowns)
                wxxx_left_node = np.matmul(Nxxx_left_node, element_unknowns)
                axial_force_left_node = self.material.E*self.material.A*ux_left_node
                shear_force_left_node = -self.material.E*self.material.I*wxxx_left_node
                bending_moment_left_node = -self.material.E*self.material.I*wxx_left_node
                f[global_element_dofs_left_node[0:1]] += axial_force_left_node
                f[global_element_dofs_left_node[1:2]] += shear_force_left_node
                f[global_element_dofs_left_node[2:3]] += bending_moment_left_node


class EulerBernoulliWeakFormDG(EulerBernoulliWeakFormCG):

    def __init__(self, function_space, material, beta):
        """
        Initialize the EulerBernoulliWeakFormDG class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
            beta: The penalty parameter for the DG method.
        """
        # invoke the parent (EulerBernoulliWeakFormCG) class
        EulerBernoulliWeakFormCG.__init__(self, function_space, material)
        # DG penalty parameter
        self.beta = beta
        # shape function values at the interface
        phi_left_interface, phixi_left_interface = self.function_space.compute_lagrange_shapes(1.0)
        N_left_interface, Nxi_left_interface, Nxixi_left_interface, Nxixixi_left_interface = \
                                    self.function_space.compute_hermite_shapes(1.0)
        phi_right_interface, phixi_right_interface = self.function_space.compute_lagrange_shapes(-1.0)
        N_right_interface, Nxi_right_interface, Nxixi_right_interface, Nxixixi_right_interface = \
                                    self.function_space.compute_hermite_shapes(-1.0)
        self.phi_left_interface = phi_left_interface
        self.phix_left_interface = phixi_left_interface*(1.0/self.function_space.jacobian)
        self.N_left_interface = N_left_interface
        self.Nx_left_interface = Nxi_left_interface*(1.0/self.function_space.jacobian)
        self.Nxx_left_interface = Nxixi_left_interface*((1.0/self.function_space.jacobian)**2.0)
        self.Nxxx_left_interface = Nxixixi_left_interface*((1.0/self.function_space.jacobian)**3.0)
        self.phi_right_interface = phi_right_interface
        self.phix_right_interface = phixi_right_interface*(1.0/self.function_space.jacobian)
        self.N_right_interface = N_right_interface
        self.Nx_right_interface = Nxi_right_interface*(1.0/self.function_space.jacobian)
        self.Nxx_right_interface = Nxixi_right_interface*((1.0/self.function_space.jacobian)**2.0)
        self.Nxxx_right_interface = Nxixixi_right_interface*((1.0/self.function_space.jacobian)**3.0)

    # Function to compute interface forces
    def __compute_interface_forces(self, element_unknowns_left, element_unknowns_right):
        """
        Compute the interface forces based on the unknowns of the left and right elements.

        Parameters:
            element_unknowns_left: The unknowns of the left element.
            element_unknowns_right: The unknowns of the right element.
        Returns:
            axial_forces_interface: The computed axial forces at the interface.
            shear_forces_interface: The computed shear forces at the interface.
            bending_moments_interface: The computed bending moments at the interface.
        """
        # dofs and their derivaitives at the interface
        # left side
        u_left = np.matmul(self.phi_left_interface, element_unknowns_left)
        axial_force_left = self.material.E*self.material.A * \
                    np.matmul(self.phix_left_interface, element_unknowns_left)
        w_left = np.matmul(self.N_left_interface, element_unknowns_left)
        wx_left = np.matmul(self.Nx_left_interface, element_unknowns_left)
        shear_force_left = -self.material.E*self.material.I * \
                                np.matmul(self.Nxxx_left_interface, element_unknowns_left)
        bending_moment_left = -self.material.E*self.material.I * \
                                np.matmul(self.Nxx_left_interface, element_unknowns_left)
        # right side
        u_right = np.matmul(self.phi_right_interface, element_unknowns_right)
        axial_force_right = self.material.E*self.material.A * \
                    np.matmul(self.phix_right_interface, element_unknowns_right)
        w_right = np.matmul(self.N_right_interface, element_unknowns_right)
        wx_right = np.matmul(self.Nx_right_interface, element_unknowns_right)
        shear_force_right = -self.material.E*self.material.I * \
                                np.matmul(self.Nxxx_right_interface, element_unknowns_right)
        bending_moment_right = -self.material.E*self.material.I * \
                                np.matmul(self.Nxx_right_interface, element_unknowns_right)
        # jumps at the interface
        u_jump = u_right - u_left
        w_jump = w_right - w_left
        wx_jump = wx_right - wx_left
        # "forces" at the interface
        axial_forces_interface = ((axial_force_left + axial_force_right) / 2.0) + \
                                        (self.beta*((self.material.E*self.material.A) / \
                                                   self.function_space.elL)*u_jump)
        shear_forces_interface = ((shear_force_left + shear_force_right) / 2.0) + \
                                        (self.beta*((self.material.E*self.material.A) / \
                                                   self.function_space.elL)*w_jump)
        bending_moments_interface = -((bending_moment_left + bending_moment_right) / 2.0) + \
                                        (self.beta*((self.material.E*self.material.I) / \
                                                   self.function_space.elL)*wx_jump)
        return axial_forces_interface, shear_forces_interface, bending_moments_interface

    # Function to compute the system residual
    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        """
        Compute the system residual based on the provided unknowns, element loads, and update flag.

        Parameters:
            f: The residual vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
            update_internal: Flag to indicate whether to update internal variables.
        """
        # compute system residual using the function in EulerBernoulliWeakFormCG
        super().compute_system_residual(f, system_unknowns, element_loads, update_internal)
        # loop over the interfaces
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # unknowns of the left and right elements
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            axial_forces_interface, shear_forces_interface, bending_moments_interface = \
                self.__compute_interface_forces(element_unknowns_left, element_unknowns_right)
            # assemble the interface forces
            f[global_element_dofs_left] += \
                    np.matmul(np.transpose(self.phi_left_interface), axial_forces_interface) + \
                    np.matmul(np.transpose(self.N_left_interface), shear_forces_interface) + \
                    np.matmul(np.transpose(self.Nx_left_interface), bending_moments_interface)
            f[global_element_dofs_right] -= \
                    np.matmul(np.transpose(self.phi_right_interface), axial_forces_interface) + \
                    np.matmul(np.transpose(self.N_right_interface), shear_forces_interface) + \
                    np.matmul(np.transpose(self.Nx_right_interface), bending_moments_interface)

    # Function to compute the system stiffness
    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        """
        Compute the system stiffness matrix based on the provided unknowns and loads.

        Parameters:
            A: The stiffness matrix to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
            element_loads: The distributed loads on the elements.
        """
        # compute system stiffness using the function in EulerBernoulliWeakFormCG
        super().compute_system_stiffness(A, system_unknowns, nodal_loads, element_loads)
        # loop over the interfaces
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # flux term tangents
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= \
                (0.50*self.material.E*self.material.A*np.matmul(np.transpose(self.phi_left_interface), \
                                                          self.phix_left_interface)) - \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.N_left_interface), \
                                                          self.Nxxx_left_interface)) + \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.Nx_left_interface), \
                                                          self.Nxx_left_interface))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= \
                (0.50*self.material.E*self.material.A*np.matmul(np.transpose(self.phi_left_interface), \
                                                          self.phix_right_interface)) - \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.N_left_interface), \
                                                          self.Nxxx_right_interface)) + \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.Nx_left_interface), \
                                                          self.Nxx_right_interface))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += \
                (0.50*self.material.E*self.material.A*np.matmul(np.transpose(self.phi_right_interface), \
                                                          self.phix_left_interface)) - \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.N_right_interface), \
                                                          self.Nxxx_left_interface)) + \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.Nx_right_interface), \
                                                          self.Nxx_left_interface))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += \
                (0.50*self.material.E*self.material.A*np.matmul(np.transpose(self.phi_right_interface), \
                                                          self.phix_right_interface)) - \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.N_right_interface), \
                                                          self.Nxxx_right_interface)) + \
                (0.50*self.material.E*self.material.I*np.matmul(np.transpose(self.Nx_right_interface), \
                                                          self.Nxx_right_interface))
            # penalty term tangents
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.phi_left_interface), self.phi_left_interface)) + \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.N_left_interface), self.N_left_interface)) + \
                (self.beta*((self.material.E*self.material.I)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.Nx_left_interface), self.Nx_left_interface))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.phi_left_interface), self.phi_right_interface)) + \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.N_left_interface), self.N_right_interface)) + \
                (self.beta*((self.material.E*self.material.I)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.Nx_left_interface), self.Nx_right_interface))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.phi_right_interface), self.phi_left_interface)) + \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.N_right_interface), self.N_left_interface)) + \
                (self.beta*((self.material.E*self.material.I)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.Nx_right_interface), self.Nx_left_interface))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.phi_right_interface), self.phi_right_interface)) + \
                (self.beta*((self.material.E*self.material.A)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.N_right_interface), self.N_right_interface)) + \
                (self.beta*((self.material.E*self.material.I)/self.function_space.elL) * \
                    np.matmul(np.transpose(self.Nx_right_interface), self.Nx_right_interface))

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
        _, phixi_left_node = self.function_space.compute_lagrange_shapes(-1.0)
        phix_left_node = phixi_left_node*(1.0/self.function_space.jacobian)
        _, _, Nxixi_left_node, Nxixixi_left_node = self.function_space.compute_hermite_shapes(-1.0)
        Nxx_left_node = Nxixi_left_node*((1.0/self.function_space.jacobian)**2.0)
        Nxxx_left_node = Nxixixi_left_node*((1.0/self.function_space.jacobian)**3.0)
        _, phixi_right_node = self.function_space.compute_lagrange_shapes(1.0)
        phix_right_node = phixi_right_node*(1.0/self.function_space.jacobian)
        _, _, Nxixi_right_node, Nxixixi_right_node = self.function_space.compute_hermite_shapes(1.0)
        Nxx_right_node = Nxixi_right_node*((1.0/self.function_space.jacobian)**2.0)
        Nxxx_right_node = Nxixixi_right_node*((1.0/self.function_space.jacobian)**3.0)
        for i in range(0, self.function_space.E): # loop over the elements
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_left_node = global_element_dofs[0:dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            # forces at the left node
            ux_left_node = np.matmul(phix_left_node, element_unknowns)
            wxx_left_node = np.matmul(Nxx_left_node, element_unknowns)
            wxxx_left_node = np.matmul(Nxxx_left_node, element_unknowns)
            axial_force_left_node = self.material.E*self.material.A*ux_left_node
            shear_force_left_node = -self.material.E*self.material.I*wxxx_left_node
            bending_moment_left_node = -self.material.E*self.material.I*wxx_left_node
            f[global_element_dofs_left_node[0:1]] += axial_force_left_node
            f[global_element_dofs_left_node[1:2]] += shear_force_left_node
            f[global_element_dofs_left_node[2:3]] += bending_moment_left_node
            # forces at the right node
            ux_right_node = np.matmul(phix_right_node, element_unknowns)
            wxx_right_node = np.matmul(Nxx_right_node, element_unknowns)
            wxxx_right_node = np.matmul(Nxxx_right_node, element_unknowns)
            axial_force_right_node = self.material.E*self.material.A*ux_right_node
            shear_force_right_node = -self.material.E*self.material.I*wxxx_right_node
            bending_moment_right_node = -self.material.E*self.material.I*wxx_right_node
            f[global_element_dofs_right_node[0:1]] += axial_force_right_node
            f[global_element_dofs_right_node[1:2]] += shear_force_right_node
            f[global_element_dofs_right_node[2:3]] += bending_moment_right_node
