import numpy as np
from beamit.WeakForm.WeakForm import WeakForm
from beamit import Material


class TFKLGeometricallyExactWeakFormCG(WeakForm):
    
    def __init__(self, function_space, material):
        """
        Initialize the TFKLGeometricallyExactWeakFormCG class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
        """
        # initialize the parent (WeakForm) class
        WeakForm.__init__(self, function_space, material)

    # Function to compute element internal forces
    def compute_element_internal_forces(self, element_unknowns):
        """
        Compute the internal forces for an element based on its unknowns.

        Parameters:
            element_unknowns: The unknowns of the element.
        Returns:
            r_el_int: The computed internal forces for the element.
        """
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npp = self.function_space.shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        rp = np.matmul(Np, element_unknowns)
        rpp = np.matmul(Npp, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dot_rpp = np.sum(rp*rpp, axis=1, keepdims=True)
        rpp_dot_rpp = np.sum(rpp*rpp, axis=1, keepdims=True)
        t1 = (rp*(rp_L2 - 1.0))/rp_L2
        t2 = (2.0*rp*((rp_dot_rpp**2.0)/(rp_L2**6.0))) - ((rp*rpp_dot_rpp)/(rp_L2**4.0)) - ((rpp*rp_dot_rpp)/(rp_L2**4.0))
        t3 = (rpp/(rp_L2**2.0)) - (rp*rp_dot_rpp)/(rp_L2**4.0)
        axial_integrand = self.material.E*self.material.A*np.matmul(np.transpose(Np, axes=(0, 2, 1)), t1)
        bending_integrand_1 = self.material.E*self.material.I*np.matmul(np.transpose(Np, axes=(0, 2, 1)), t2)
        bending_integrand_2 = self.material.E*self.material.I*np.matmul(np.transpose(Npp, axes=(0, 2, 1)), t3)
        r_el_int = np.sum((axial_integrand + bending_integrand_1 + bending_integrand_2)*self.function_space.JxW, axis=0, keepdims=False)
        return r_el_int

    # Function to compute element internal stiffness
    def compute_element_internal_stiffness(self, element_unknowns):
        """
        Compute the internal stiffness for an element based on its unknowns.

        Parameters:
            element_unknowns: The unknowns of the element.
        Returns:
            K_el_int: The computed internal stiffness matrix for the element.
        """
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npp = self.function_space.shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        rp = np.matmul(Np, element_unknowns)
        rpp = np.matmul(Npp, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dot_rpp = np.sum(rp*rpp, axis=1, keepdims=True)
        rpp_dot_rpp = np.sum(rpp*rpp, axis=1, keepdims=True)
        rp_dyd_rp = np.matmul(rp, np.transpose(rp, axes=(0, 2, 1)))
        rp_dyd_rpp = np.matmul(rp, np.transpose(rpp, axes=(0, 2, 1)))
        rpp_dyd_rp = np.matmul(rpp, np.transpose(rp, axes=(0, 2, 1)))
        rpp_dyd_rpp = np.matmul(rpp, np.transpose(rpp, axes=(0, 2, 1)))
        imat = np.eye(self.function_space.dim)
        dt1dd = np.matmul(((((rp_L2 - 1.0)/rp_L2)*imat) + ((1.0/(rp_L2**3.0))*rp_dyd_rp)), Np)
        dt2dd_term1 = ((2.0*((rp_dot_rpp**2.0)/(rp_L2**6.0))) - ((rpp_dot_rpp)/(rp_L2**4.0)))*imat
        dt2dd_term2 = ((-12.0*((rp_dot_rpp**2.0)/(rp_L2**8.0))) + (4.0*((rpp_dot_rpp)/(rp_L2**6.0))))*rp_dyd_rp
        dt2dd_term3 = (4.0*((rp_dot_rpp)/(rp_L2**6.0)))*(rp_dyd_rpp + rpp_dyd_rp)
        dt2dd_term4 = (1.0/(rp_L2**4.0))*rpp_dyd_rpp
        dt2dd_term5 = -(((rp_dot_rpp)/(rp_L2**4.0))*imat) + ((4.0*((rp_dot_rpp)/(rp_L2**6.0)))*rp_dyd_rp)
        dt2dd_term6 = ((2.0/(rp_L2**4.0))*rp_dyd_rpp) + ((1.0/(rp_L2**4.0))*rpp_dyd_rp)
        dt2dd = np.matmul((dt2dd_term1 + dt2dd_term2 + dt2dd_term3 - dt2dd_term4), Np) + \
                                    np.matmul((dt2dd_term5 - dt2dd_term6), Npp)
        dt3dd_term1 = dt2dd_term5
        dt3dd_term2 = ((2.0/(rp_L2**4.0))*rpp_dyd_rp) + ((1.0/(rp_L2**4.0))*rp_dyd_rpp)
        dt3dd_term3 = ((1.0/(rp_L2**2.0))*imat) - ((1.0/(rp_L2**4.0))*rp_dyd_rp) 
        dt3dd = np.matmul((dt3dd_term1 - dt3dd_term2), Np) + np.matmul((dt3dd_term3), Npp)
        axial_integrand = self.material.E*self.material.A*np.matmul(np.transpose(Np, axes=(0, 2, 1)), dt1dd)
        bending_integrand_1 = self.material.E*self.material.I*np.matmul(np.transpose(Np, axes=(0, 2, 1)), dt2dd)
        bending_integrand_2 = self.material.E*self.material.I*np.matmul(np.transpose(Npp, axes=(0, 2, 1)), dt3dd)
        K_el_int = np.sum((axial_integrand + bending_integrand_1 + bending_integrand_2)*self.function_space.JxW, axis=0, keepdims=False)
        return K_el_int

    # Function to compute element external distributed forces (for distributed forces and moments)
    def compute_element_external_distributed_forces(self, element_unknowns, el_dist_forces, el_dist_moments):
        """
        Compute the external distributed forces and moments for an element based on its unknowns.

        Parameters:
            element_unknowns: The unknowns of the element.
            el_dist_forces: The distributed forces on the element.
            el_dist_moments: The distributed moments on the element.
        Returns:
            r_el_dist: The computed residual for the distributed forces and moments.
        """
        Nt = np.transpose(self.function_space.shape_functions, axes=(0, 2, 1))
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npt = np.transpose(Np, axes=(0, 2, 1))
        rp = np.matmul(Np, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        t4 = rp/(rp_L2**2.0)
        force_integrand = np.matmul(Nt, el_dist_forces)
        r_el_dist_forces = np.sum(force_integrand*self.function_space.JxW, axis=0, keepdims=False)
        mdist_cross_t4 = np.cross(el_dist_moments, t4, axisa=1, axisb=0, axisc=1)
        moment_integrand = np.matmul(Npt, mdist_cross_t4)
        r_el_dist_moments = np.sum(moment_integrand*self.function_space.JxW, axis=0, keepdims=False)
        return r_el_dist_forces + r_el_dist_moments

    # Function to add nodal loads to the residual
    def add_nodal_loads_to_residual(self, f, system_unknowns, nodal_loads):
        """
        Add the nodal loads to the residual vector.

        Parameters:
            f: The residual vector to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
        """
        # update the contribution of external nodal moments to the residual
        dofs = self.function_space.dof
        updated_nodal_loads = np.zeros(nodal_loads.shape)
        # loop over the nodes
        for i in range(0, self.function_space.N):
            updated_nodal_loads[dofs*i:(dofs*i)+3, :] += nodal_loads[dofs*i:(dofs*i)+3, :]
            nodal_tangents = system_unknowns[(dofs*i)+3:(dofs*i)+6, :]
            nodal_tangents_L2 = np.linalg.norm(nodal_tangents, ord=2, axis=0, keepdims=True)
            t4_nodal = nodal_tangents/(nodal_tangents_L2**2.0)
            # compute the cross-product for the nodal moments
            updated_nodal_loads[(dofs*i)+3:(dofs*i)+6, :] += np.cross(
                nodal_loads[(dofs*i)+3:(dofs*i)+6, :], t4_nodal, axis=0)
        f += updated_nodal_loads

    # Function to compute the system stiffness matrix
    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        """
        Compute the system stiffness matrix based on the provided unknowns and element loads information.

        Parameters:
            A: The stiffness matrix to be assembled.
            system_unknowns: The unknowns of the system.
            nodal_loads: The nodal loads applied to the system.
            element_loads: The distributed loads on the elements.
        """
        super().compute_system_stiffness(A, system_unknowns, nodal_loads, element_loads)
        # add the contribution of external nodal moments to the system stiffness
        dofs = self.function_space.dof
        # loop over the nodes
        for i in range(0, self.function_space.N):
            nodal_tangents = system_unknowns[(dofs*i)+3:(dofs*i)+6, :]
            nodal_tangents_L2 = np.linalg.norm(nodal_tangents, ord=2, axis=0, keepdims=True)
            nodal_moments = nodal_loads[(dofs*i)+3:(dofs*i)+6, :]
            nodal_rp_dyd_rp = np.matmul(nodal_tangents, np.transpose(nodal_tangents))
            nodal_dt4dd_coeff = (np.eye(self.function_space.dim)/(nodal_tangents_L2**2.0)) - (
                (2.0*nodal_rp_dyd_rp)/(nodal_tangents_L2**4.0))
            A[(dofs*i)+3:(dofs*i)+6, (dofs*i)+3:(dofs*i) +
              6] -= np.cross(nodal_moments, nodal_dt4dd_coeff, axis=0)

    @staticmethod
    def _compute_residual_vectors(rp, rpp, rppp):
        """
        Compute the 't_i' vectors in the residual.

        Parameters:
            rp: The first derivative of the position vector.
            rpp: The second derivative of the position vector.
            rppp: The third derivative of the position vector.
        Returns:
            t1, t2, t3, t4, t5: The computed 't_i' vectors.
        """
        rp_L2 = np.linalg.norm(rp, ord=2, axis=0, keepdims=True)
        rp_dot_rpp = np.sum(rp*rpp, axis=0, keepdims=True)
        rpp_dot_rpp = np.sum(rpp*rpp, axis=0, keepdims=True)
        rp_dot_rppp = np.sum(rp*rppp, axis=0, keepdims=True)
        t1 = (rp*(rp_L2 - 1.0))/rp_L2
        t2 = (2.0*rp*((rp_dot_rpp**2.0)/(rp_L2**6.0))) - ((rp*rpp_dot_rpp)/(rp_L2**4.0)) - ((rpp*rp_dot_rpp)/(rp_L2**4.0))
        t3 = (rpp/(rp_L2**2.0)) - (rp*rp_dot_rpp)/(rp_L2**4.0)
        t4 = rp/(rp_L2**2.0)
        t5 = ((2.0*rpp*rp_dot_rpp)/(rp_L2**4.0)) - ((2.0*rp*(rp_dot_rpp**2.0))/(rp_L2**6.0)) + ((rp*rp_dot_rppp)/(rp_L2**4.0)) - (rppp/(rp_L2**2.0))
        return t1, t2, t3, t4, t5

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
        _, Nxi_left_node, Nxixi_left_node, Nxixixi_left_node = self.function_space.compute_shapes(-1.0)
        Np_left_node = Nxi_left_node*(1.0/self.function_space.jacobian)
        Npp_left_node = Nxixi_left_node*((1.0/self.function_space.jacobian)**2.0)
        Nppp_left_node = Nxixixi_left_node*((1.0/self.function_space.jacobian)**3.0)
        _, Nxi_right_node, Nxixi_right_node, Nxixixi_right_node = self.function_space.compute_shapes(1.0)
        Np_right_node = Nxi_right_node*(1.0/self.function_space.jacobian)
        Npp_right_node = Nxixi_right_node*((1.0/self.function_space.jacobian)**2.0)
        Nppp_right_node = Nxixixi_right_node*((1.0/self.function_space.jacobian)**3.0)
        for i in range(0, self.function_space.E): # loop over the elements
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_left_node = global_element_dofs[0:dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            rp_left_node = np.matmul(Np_left_node, element_unknowns)
            rp_right_node = np.matmul(Np_right_node, element_unknowns)
            rpp_left_node = np.matmul(Npp_left_node, element_unknowns)
            rpp_right_node = np.matmul(Npp_right_node, element_unknowns)
            rppp_left_node = np.matmul(Nppp_left_node, element_unknowns)
            rppp_right_node = np.matmul(Nppp_right_node, element_unknowns)
            rp_left_node_L2 = np.linalg.norm(rp_left_node, ord=2, axis=0, keepdims=True)
            rp_right_node_L2 = np.linalg.norm(rp_right_node, ord=2, axis=0, keepdims=True)
            t1_left_node, _, _, _, t5_left_node = self._compute_residual_vectors(rp_left_node, rpp_left_node, rppp_left_node)
            t1_right_node, _, _, _, t5_right_node = self._compute_residual_vectors(rp_right_node, rpp_right_node, rppp_right_node)
            # forces at the nodes
            if (element_loads == None): # No element loads
                forces_left_node = (self.material.E*self.material.A*t1_left_node) + (self.material.E*self.material.I*t5_left_node)
                forces_right_node = (self.material.E*self.material.A*t1_right_node) + (self.material.E*self.material.I*t5_right_node)
            # moments at the nodes
            moments_left_node = self.material.E*self.material.I*(np.cross(rp_left_node, rpp_left_node, axis=0)/(rp_left_node_L2**2.0))
            moments_right_node = self.material.E*self.material.I*(np.cross(rp_right_node, rpp_right_node, axis=0)/(rp_right_node_L2**2.0))
            f[global_element_dofs_left_node[0:int(dofs/2)]] += forces_left_node
            f[global_element_dofs_left_node[int(dofs/2):dofs]] += moments_left_node
            f[global_element_dofs_right_node[0:int(dofs/2)]] += forces_right_node
            f[global_element_dofs_right_node[int(dofs/2):dofs]] += moments_right_node

    # Function to compute the system mass
    def compute_system_mass(self, M, lump=True, **kwargs):
        """
        Compute the system mass matrix.

        Parameters:
            M: The mass matrix to be assembled.
            lump: If True, apply lumping to the mass matrix.
            **kwargs: Optional keyword arguments.
        """
        Nt = np.transpose(self.function_space.shape_functions, axes=(0, 2, 1))
        translational_mass_integrand = self.material.rho*self.material.A*np.matmul(Nt, self.function_space.shape_functions)
        # the element translational mass matrix
        M_el_trans = np.sum(translational_mass_integrand*self.function_space.JxW, axis=0, keepdims=False)
        # apply "special lumping" to the element translational mass matrix
        if (lump):
            sum_all_entries = np.sum(M_el_trans)
            sum_diag_entries = np.sum(np.diag(M_el_trans))
            diag_elements = (sum_all_entries / sum_diag_entries) * np.diag(M_el_trans)
            M_el_trans = np.zeros([self.function_space.npel*self.function_space.dof,
                                  self.function_space.npel*self.function_space.dof])
            np.fill_diagonal(M_el_trans, diag_elements)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            M[np.ix_(global_element_dofs, global_element_dofs)] += M_el_trans


class TFKLGeometricallyExactWeakFormDG(TFKLGeometricallyExactWeakFormCG):

    def __init__(self, function_space, material, betaP, betaT):
        """
        Initialize the TFKLGeometricallyExactWeakFormDG class.

        Parameters:
            function_space: The function space containing the geometrical information.
            material: The material properties containing the physical information.
            betaP: The penalty parameter for the position jump.
            betaT: The penalty parameter for the tangent jump.
        """
        # invoke the parent (TFKLGeometricallyExactWeakFormCG) class
        TFKLGeometricallyExactWeakFormCG.__init__(
            self, function_space, material)
        # DG position jump penalty parameter
        self.betaP = betaP
        # DG tangent jump penalty parameter
        self.betaT = betaT
        # store internal variables (of CZM) at all the interfaces
        # first column = binary parameter to indicate damage status at the interface
        # (0.0 -> damage 'not' initiated, 1.0 -> damage initiated)
        # second column = binary parameter to enable/disable DG flux and compatibility, and CZM force terms in the interface jump forces
        # (0.0 -> DG flux and compatibility terms are active, 1.0 -> CZM force terms are active)
        # third column = maximum effective separation at an interface in the entire loading history
        # fourth column = effective force at an interface at damage initiation
        # fifth to seventh columns = position jumps at an interface at damage initiation
        # eighth to tenth columns = tangent jumps at an interface at damage initiation
        # eleventh column = binary parameter to switch between axial DG and CZM terms in case of recontact at an interface "after damage initiation"
        self.internal_variables = np.zeros([self.function_space.E-1, 11])

    # Function to compute interface forces
    def __compute_interface_forces(self, element_unknowns_left, element_unknowns_right, 
                                   element_internal_variables, element_loads, update_internal):
        """
        Compute the interface forces and moments between two elements based on their unknowns.

        Parameters:
            element_unknowns_left: The unknowns of the left element.
            element_unknowns_right: The unknowns of the right element.
            element_internal_variables: The internal variables at the interface.
            element_loads: The distributed loads on the elements.
            update_internal: If True, update the internal variables in the weak form.
        Returns:
            average_forces_interface: The average forces at the interface.
            average_mxt4_interface: The average bending moments at the interface.
            cohesive_forces: The cohesive forces at the interface (if applicable).
            cohesive_bending_moments: The cohesive bending moments at the interface (if applicable).
        """
        # shape functions derivatives at the interfaces (left (-) & right (+))
        N_left_interface, Nxi_left_interface, Nxixi_left_interface, Nxixixi_left_interface = \
            self.function_space.compute_shapes(1.0)
        Np_left_interface = Nxi_left_interface * (1.0/self.function_space.jacobian)
        Npp_left_interface = Nxixi_left_interface * ((1.0/self.function_space.jacobian)**2.0)
        Nppp_left_interface = Nxixixi_left_interface * ((1.0/self.function_space.jacobian)**3.0)
        N_right_interface, Nxi_right_interface, Nxixi_right_interface, Nxixixi_right_interface = \
            self.function_space.compute_shapes(-1.0)
        Np_right_interface = Nxi_right_interface * (1.0/self.function_space.jacobian)
        Npp_right_interface = Nxixi_right_interface * ((1.0/self.function_space.jacobian)**2.0)
        Nppp_right_interface = Nxixixi_right_interface * ((1.0/self.function_space.jacobian)**3.0)
        # parameterization and its derivatives at the interface
        r_left_interface = np.matmul(N_left_interface, element_unknowns_left)
        r_right_interface = np.matmul(N_right_interface, element_unknowns_right)
        rp_left_interface = np.matmul(Np_left_interface, element_unknowns_left)
        rp_right_interface = np.matmul(Np_right_interface, element_unknowns_right)
        rpp_left_interface = np.matmul(Npp_left_interface, element_unknowns_left)
        rpp_right_interface = np.matmul(Npp_right_interface, element_unknowns_right)
        rppp_left_interface = np.matmul(Nppp_left_interface, element_unknowns_left)
        rppp_right_interface = np.matmul(Nppp_right_interface, element_unknowns_right)
        rp_left_interface_L2 = np.linalg.norm(rp_left_interface, ord=2, axis=0, keepdims=True)
        rp_right_interface_L2 = np.linalg.norm(rp_right_interface, ord=2, axis=0, keepdims=True)
        t1_left_interface, _, _, t4_left_interface, t5_left_interface = self._compute_residual_vectors(rp_left_interface, rpp_left_interface, rppp_left_interface)
        t1_right_interface, _, _, t4_right_interface, t5_right_interface = self._compute_residual_vectors(rp_right_interface, rpp_right_interface, rppp_right_interface)
        # forces at the interface
        if (element_loads == None): # No element loads
            forces_left_interface = (self.material.E*self.material.A*t1_left_interface) + (self.material.E*self.material.I*t5_left_interface)
            forces_right_interface = (self.material.E*self.material.A*t1_right_interface) + (self.material.E*self.material.I*t5_right_interface)
        average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
        # moments at the interface
        moments_left_interface = self.material.E*self.material.I*(np.cross(rp_left_interface, rpp_left_interface, axis=0)/(rp_left_interface_L2**2.0))
        moments_right_interface = self.material.E*self.material.I*(np.cross(rp_right_interface, rpp_right_interface, axis=0)/(rp_right_interface_L2**2.0))
        mxt4_left_interface = np.cross(moments_left_interface, t4_left_interface, axis=0)
        mxt4_right_interface = np.cross(moments_right_interface, t4_right_interface, axis=0)
        average_mxt4_interface = (mxt4_left_interface + mxt4_right_interface)/2.0
        # compute the cohesive forces and bending moments at the interface
        cohesive_forces, cohesive_bending_moments = self.__compute_CZM_interface_forces(
            r_left_interface, r_right_interface, 
            rp_left_interface, rp_right_interface, 
            forces_left_interface, forces_right_interface, 
            moments_left_interface, moments_right_interface, 
            element_internal_variables, update_internal)
        return average_forces_interface, average_mxt4_interface, cohesive_forces, cohesive_bending_moments

    def __compute_CZM_interface_forces(self, r_left_interface, r_right_interface, 
                                       rp_left_interface, rp_right_interface, forces_left_interface, 
                                       forces_right_interface, moments_left_interface, 
                                       moments_right_interface, element_internal_variables, 
                                       update_internal):
        """
        Compute the cohesive zone model (CZM) interface forces and moments.

        Parameters:
            r_left_interface: The position vector at the left interface.
            r_right_interface: The position vector at the right interface.
            rp_left_interface: The first derivative of the position vector at the left interface.
            rp_right_interface: The first derivative of the position vector at the right interface.
            forces_left_interface: The forces at the left interface.
            forces_right_interface: The forces at the right interface.
            moments_left_interface: The moments at the left interface.
            moments_right_interface: The moments at the right interface.
            element_internal_variables: The internal variables at the interface.
            update_internal: If True, update the internal variables in the weak form.
        Returns:
            cohesive_forces: The cohesive forces at the interface (if applicable).
            cohesive_bending_moments: The cohesive bending moments at the interface (if applicable).
        """
        # initialize cohesive forces and bending moments
        cohesive_forces = np.zeros([self.function_space.dim, 1])
        cohesive_bending_moments = np.zeros([self.function_space.dim, 1])
        # perform CZM checks and calculations in the case of a cohesive interface material
        if ((isinstance(self.material, (Material.TFKLCohesiveInterfaceMaterial))) and update_internal):
            # position and tangent jumps at the interface
            r_jump_interface = r_right_interface - r_left_interface
            rp_jump_interface = rp_right_interface - rp_left_interface
            # average forces at the interface
            average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
            # just after damage initiation or damage not yet initiated
            if (element_internal_variables[0:1, 2:3] == 0.0):
                # just after damage initiation
                if (element_internal_variables[0:1, 0:1] == 1.0):
                    # effective separation at the interface
                    delta = self.material.compute_effective_separation(r_left_interface, 
                                                                       r_right_interface, 
                                                                       rp_left_interface, 
                                                                       rp_right_interface,
                                                                       position_jumps_DI=element_internal_variables[0:1, 4:7].T, tangent_jumps_DI=element_internal_variables[0:1, 7:10].T)
                    if (delta == 0.0):  # fall back to DG terms
                        element_internal_variables[0:1, 1:2] = 0.0
                    else:  # perform CZM calculations
                        element_internal_variables[0:1, 1:2] = 1.0
                        axial_jump = self.material.compute_axial_separation(
                            r_left_interface, r_right_interface, rp_left_interface, 
                            rp_right_interface, position_jumps_DI=element_internal_variables[0:1, 4:7].T)
                        if (axial_jump < 0.0):  # recontact at the interface - NOT USING THIS ONE FOR NOW!!!
                            element_internal_variables[0:1, 10:11] = 0.0
                        else:
                            element_internal_variables[0:1, 10:11] = 1.0
                        # evaluate cohesive forces according to the TSL
                        cohesive_axial_forces = \
                            self.material.compute_cohesive_axial_forces(r_left_interface, 
                                                                        r_right_interface, 
                                                                        rp_left_interface, 
                                                                        rp_right_interface, 
                                                                        delta_max=element_internal_variables[0:1, 2:3], 
                                                                        position_jumps_DI=element_internal_variables[0:1, 4:7].T, 
                                                                        tangent_jumps_DI=element_internal_variables[0:1, 7:10].T)
                        # compute the cohesive forces and bending moments
                        # for complete damage
                        if ((delta >= self.material.delta_c) or 
                            (element_internal_variables[0:1, 2:3] == self.material.delta_c)):
                            interface_constrained_shear_forces = np.zeros(
                                average_forces_interface.shape)
                        else:
                            effective_unit_tangent_interface = self.material.compute_effective_unit_tangent(
                                rp_left_interface, rp_right_interface)
                            tangeff_dyd_tangeff = np.matmul(
                                effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
                            interface_constrained_shear_forces = np.matmul((np.eye(3)-tangeff_dyd_tangeff), average_forces_interface) + (self.betaP*(
                                (self.material.E*self.material.A)/self.function_space.elL)*np.matmul((np.eye(3)-tangeff_dyd_tangeff), r_jump_interface))
                        cohesive_forces = cohesive_axial_forces + interface_constrained_shear_forces
                        cohesive_bending_moments = \
                            self.material.compute_cohesive_bending_moments(r_left_interface, 
                                                                           r_right_interface, 
                                                                           rp_left_interface, 
                                                                           rp_right_interface, 
                                                                           delta_max=element_internal_variables[0:1, 2:3],
                                                                           position_jumps_DI=element_internal_variables[0:1, 4:7].T, tangent_jumps_DI=element_internal_variables[0:1, 7:10].T)
                        if (update_internal):
                            # update the maximum effective separation
                            new_delta_max = self.material.compute_effective_maximum_separation(
                                delta, delta_max=element_internal_variables[0:1, 2:3])
                            element_internal_variables[0:1, 2:3] = new_delta_max
                # damage not yet initiated
                # evaluate the damage initiation criterion
                elif (self.material.evaluate_damage_initiation_criterion(rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface)):
                    # damage just initiated at the interface
                    element_internal_variables[0:1, 0:1] = 1.0
                    # keep the DG flux and compatibility terms active immediately after damage initiation (since delta = 0.0)
                    element_internal_variables[0:1, 1:2] = 0.0
                    # update the internal variables at damage initiation
                    element_internal_variables[0:1, 3:4] = self.material.compute_effective_force(
                        rp_left_interface, rp_right_interface, forces_left_interface, 
                        forces_right_interface, moments_left_interface, moments_right_interface)
                    element_internal_variables[0:1, 4:7] = r_jump_interface.T
                    element_internal_variables[0:1, 7:10] = rp_jump_interface.T
                else:  # no damage at the interface
                    element_internal_variables[0:1, 1:2] = 0.0
            # damage already initiated at the interface (loading | unloading | damage after recontact)
            else:
                # effective separation at the interface
                delta = self.material.compute_effective_separation(r_left_interface, 
                                                                   r_right_interface, 
                                                                   rp_left_interface, 
                                                                   rp_right_interface,
                                                                   position_jumps_DI=element_internal_variables[0:1, 4:7].T, 
                                                                   tangent_jumps_DI=element_internal_variables[0:1, 7:10].T)
                if (delta == 0.0):  # fall back to DG terms
                    element_internal_variables[0:1, 1:2] = 0.0
                else:  # perform CZM calculations
                    element_internal_variables[0:1, 1:2] = 1.0
                    axial_jump = self.material.compute_axial_separation(
                        r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, 
                        position_jumps_DI=element_internal_variables[0:1, 4:7].T)
                    if (axial_jump < 0.0):  # recontact at the interface - NOT USING THIS ONE FOR NOW!!!
                        element_internal_variables[0:1, 10:11] = 0.0
                    else:
                        element_internal_variables[0:1, 10:11] = 1.0
                    # evaluate cohesive forces according to the TSL
                    cohesive_axial_forces = \
                        self.material.compute_cohesive_axial_forces(r_left_interface, 
                                                                    r_right_interface, 
                                                                    rp_left_interface, 
                                                                    rp_right_interface, 
                                                                    delta_max=element_internal_variables[0:1, 2:3], 
                                                                    position_jumps_DI=element_internal_variables[0:1, 4:7].T, tangent_jumps_DI=element_internal_variables[0:1, 7:10].T)
                    # compute the cohesive forces and bending moments
                    # for complete damage
                    if ((delta >= self.material.delta_c) or (element_internal_variables[0:1, 2:3] == self.material.delta_c)):
                        interface_constrained_shear_forces = np.zeros(
                            average_forces_interface.shape)
                    else:
                        effective_unit_tangent_interface = self.material.compute_effective_unit_tangent(
                            rp_left_interface, rp_right_interface)
                        tangeff_dyd_tangeff = np.matmul(
                            effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
                        interface_constrained_shear_forces = np.matmul((np.eye(3)-tangeff_dyd_tangeff), average_forces_interface) + (self.betaP*(
                            (self.material.E*self.material.A)/self.function_space.elL)*np.matmul((np.eye(3)-tangeff_dyd_tangeff), r_jump_interface))
                    cohesive_forces = cohesive_axial_forces + interface_constrained_shear_forces
                    cohesive_bending_moments = \
                        self.material.compute_cohesive_bending_moments(r_left_interface, 
                                                                       r_right_interface, 
                                                                       rp_left_interface, 
                                                                       rp_right_interface, 
                                                                       delta_max=element_internal_variables[0:1, 2:3],
                                                                       position_jumps_DI=element_internal_variables[0:1, 4:7].T, tangent_jumps_DI=element_internal_variables[0:1, 7:10].T)
                    if (update_internal):
                        # update the maximum effective separation
                        new_delta_max = self.material.compute_effective_maximum_separation(
                            delta, delta_max=element_internal_variables[0:1, 2:3])
                        element_internal_variables[0:1, 2:3] = new_delta_max
        return cohesive_forces, cohesive_bending_moments

    # Function to compute the system residual
    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        """
        Compute the system residual based on the provided unknowns and element loads information.

        Parameters:
            f: The residual vector to be assembled.
            system_unknowns: The unknowns of the system.
            element_loads: The distributed loads on the elements.
            update_internal: If True, update the internal variables in the weak form.
        """
        # compute system residual using the function in the parent class
        super().compute_system_residual(f, system_unknowns, element_loads, update_internal)
        # add the contributions of jump terms at the interfaces to the residual
        # shape functions and their derivatives at the interfaces (left (-) & right (+))
        N_left_interface, Nxi_left_interface, _, _ = self.function_space.compute_shapes(1.0)
        Np_left_interface = Nxi_left_interface*(1.0/self.function_space.jacobian)
        N_right_interface, Nxi_right_interface, _, _ = self.function_space.compute_shapes(-1.0)
        Np_right_interface = Nxi_right_interface*(1.0/self.function_space.jacobian)
        # loop over the interfaces
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # unknowns of the left and right elements
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            # parameterization and its derivatives at the interface
            r_left_interface = np.matmul(N_left_interface, element_unknowns_left)
            r_right_interface = np.matmul(N_right_interface, element_unknowns_right)
            rp_left_interface = np.matmul(Np_left_interface, element_unknowns_left)
            rp_right_interface = np.matmul(Np_right_interface, element_unknowns_right)
            # position and tangent jumps at the interface
            r_jump_interface = r_right_interface - r_left_interface
            rp_jump_interface = rp_right_interface - rp_left_interface
            # compute the interface forces and moments
            average_forces_interface, average_mxt4_interface, cohesive_forces, \
                cohesive_bending_moments = self.__compute_interface_forces(
                    element_unknowns_left, element_unknowns_right, 
                    self.internal_variables[i:i+1, :], element_loads, update_internal)
            # DG FLUX AND COMPATIBILITY TERMS
            f[global_element_dofs_left] += (1.0-self.internal_variables[i:i+1, 1:2])*(np.matmul(np.transpose(N_left_interface), average_forces_interface) + np.matmul(np.transpose(Np_left_interface), average_mxt4_interface) + (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), r_jump_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), rp_jump_interface)))
            f[global_element_dofs_right] -= (1.0-self.internal_variables[i:i+1, 1:2])*(np.matmul(np.transpose(N_right_interface), average_forces_interface) + np.matmul(np.transpose(Np_right_interface), average_mxt4_interface) + (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), r_jump_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), rp_jump_interface)))
            # INTERFACE FORCES AND MOMENTS FROM THE CZM
            f[global_element_dofs_left] += (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(N_left_interface), cohesive_forces)))
            f[global_element_dofs_right] -= (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(N_right_interface), cohesive_forces)))
            f[global_element_dofs_left] += (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(Np_left_interface), cohesive_bending_moments)))
            f[global_element_dofs_right] -= (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(Np_right_interface), cohesive_bending_moments)))

    @staticmethod
    def _compute_jump_stiffness_coefficients(rp, rpp, rppp):
        """
        Computes the coefficients of the 't_i' vector gradients in the jump stiffness.
        Parameters:
            rp (np.ndarray): Position vector.
            rpp (np.ndarray): First derivative of the position vector.
            rppp (np.ndarray): Second derivative of the position vector.
        Returns:
            tuple: Coefficients dt1dd_Np, dt3dd_Np, dt3dd_Npp, dt5dd_Np, dt5dd_Npp, dt5dd_Nppp.
        """
        imat = np.eye(rp.shape[0])
        rp_L2 = np.linalg.norm(rp, ord=2, axis=0, keepdims=True)
        rp_dot_rpp = np.sum(rp*rpp, axis=0, keepdims=True)
        rp_dot_rppp = np.sum(rp*rppp, axis=0, keepdims=True)
        rp_dyd_rp = np.matmul(rp, np.transpose(rp))
        rp_dyd_rpp = np.matmul(rp, np.transpose(rpp))
        rpp_dyd_rp = np.matmul(rpp, np.transpose(rp))
        rpp_dyd_rpp = np.matmul(rpp, np.transpose(rpp))
        rp_dyd_rppp = np.matmul(rp, np.transpose(rppp))
        rppp_dyd_rp = np.matmul(rppp, np.transpose(rp))
        dt1dd_Np = (((rp_L2 - 1.0)/rp_L2)*imat) + ((1.0/(rp_L2**3.0))*rp_dyd_rp)
        dt3dd_term1 = -(((rp_dot_rpp)/(rp_L2**4.0))*imat) + ((4.0*((rp_dot_rpp)/(rp_L2**6.0)))*rp_dyd_rp)
        dt3dd_term2 = ((2.0/(rp_L2**4.0))*rpp_dyd_rp) + ((1.0/(rp_L2**4.0))*rp_dyd_rpp)
        dt3dd_term3 = ((1.0/(rp_L2**2.0))*imat) - ((1.0/(rp_L2**4.0))*rp_dyd_rp)
        dt3dd_Np = dt3dd_term1 - dt3dd_term2
        dt3dd_Npp = dt3dd_term3
        dt5dd_term1 = ((-2.0*((rp_dot_rpp**2.0)/(rp_L2**6.0))) + ((rp_dot_rppp)/(rp_L2**4.0)))*imat
        dt5dd_term2 = ((12.0*((rp_dot_rpp**2.0)/(rp_L2**8.0))) - (4.0*((rp_dot_rppp)/(rp_L2**6.0))))*rp_dyd_rp
        dt5dd_term3 = (4.0*((rp_dot_rpp)/(rp_L2**6.0)))*(rp_dyd_rpp + (2.0*rpp_dyd_rp))
        dt5dd_term4 = ((2.0*rpp_dyd_rpp)/(rp_L2**4.0)) + (rp_dyd_rppp/(rp_L2**4.0)) + ((2.0*rppp_dyd_rp)/(rp_L2**4.0))
        dt5dd_Np = dt5dd_term1 + dt5dd_term2 - dt5dd_term3 + dt5dd_term4
        dt5dd_Npp = ((2.0*((rp_dot_rpp)/(rp_L2**4.0)))*imat) - ((4.0*((rp_dot_rpp)/(rp_L2**6.0)))*rp_dyd_rp) + ((2.0*rpp_dyd_rp)/(rp_L2**4.0))
        dt5dd_Nppp = -((1.0/(rp_L2**2.0))*imat) + ((1.0/(rp_L2**4.0))*rp_dyd_rp)
        return dt1dd_Np, dt3dd_Np, dt3dd_Npp, dt5dd_Np, dt5dd_Npp, dt5dd_Nppp

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
        # compute system stiffness using the function in the parent class
        super().compute_system_stiffness(A, system_unknowns, nodal_loads, element_loads)
        # add the contributions of jump terms at the interfaces to the residual
        # shape functions and their derivatives at the interfaces (left (-) & right (+))
        N_left_interface, Nxi_left_interface, Nxixi_left_interface, Nxixixi_left_interface = self.function_space.compute_shapes(1.0)
        Np_left_interface = Nxi_left_interface*(1.0/self.function_space.jacobian)
        Npp_left_interface = Nxixi_left_interface*((1.0/self.function_space.jacobian)**2.0)
        Nppp_left_interface = Nxixixi_left_interface*((1.0/self.function_space.jacobian)**3.0)
        N_right_interface, Nxi_right_interface, Nxixi_right_interface, Nxixixi_right_interface = self.function_space.compute_shapes(-1.0)
        Np_right_interface = Nxi_right_interface*(1.0/self.function_space.jacobian)
        Npp_right_interface = Nxixi_right_interface*((1.0/self.function_space.jacobian)**2.0)
        Nppp_right_interface = Nxixixi_right_interface*((1.0/self.function_space.jacobian)**3.0)
        # loop over the interfaces
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # unknowns of the left and right elements
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            # parameterizations at the interface
            rp_left_interface = np.matmul(Np_left_interface, element_unknowns_left)
            rp_right_interface = np.matmul(Np_right_interface, element_unknowns_right)
            rpp_left_interface = np.matmul(Npp_left_interface, element_unknowns_left)
            rpp_right_interface = np.matmul(Npp_right_interface, element_unknowns_right)
            rppp_left_interface = np.matmul(Nppp_left_interface, element_unknowns_left)
            rppp_right_interface = np.matmul(Nppp_right_interface, element_unknowns_right)
            ############## flux and compatibility term derivatives ##############
            # coefficients of 't_i' vector gradients at the interface
            dt1dd_Np_left_interface, dt3dd_Np_left_interface, dt3dd_Npp_left_interface, dt5dd_Np_left_interface, dt5dd_Npp_left_interface, dt5dd_Nppp_left_interface = self._compute_jump_stiffness_coefficients(
                rp_left_interface, rpp_left_interface, rppp_left_interface)
            dt1dd_Np_right_interface, dt3dd_Np_right_interface, dt3dd_Npp_right_interface, dt5dd_Np_right_interface, dt5dd_Npp_right_interface, dt5dd_Nppp_right_interface = self._compute_jump_stiffness_coefficients(
                rp_right_interface, rpp_right_interface, rppp_right_interface)
            # 't_i' vector gradients at the interface
            dt1dd_left_interface = np.matmul(dt1dd_Np_left_interface, Np_left_interface)
            dt1dd_right_interface = np.matmul(dt1dd_Np_right_interface, Np_right_interface)
            dt3dd_left_interface = np.matmul(dt3dd_Np_left_interface, Np_left_interface) + np.matmul(dt3dd_Npp_left_interface, Npp_left_interface)
            dt3dd_right_interface = np.matmul(dt3dd_Np_right_interface, Np_right_interface) + np.matmul(dt3dd_Npp_right_interface, Npp_right_interface)
            dt5dd_left_interface = np.matmul(dt5dd_Np_left_interface, Np_left_interface) + np.matmul(dt5dd_Npp_left_interface, Npp_left_interface) + np.matmul(dt5dd_Nppp_left_interface, Nppp_left_interface)
            dt5dd_right_interface = np.matmul(dt5dd_Np_right_interface, Np_right_interface) + np.matmul(dt5dd_Npp_right_interface, Npp_right_interface) + np.matmul(dt5dd_Nppp_right_interface, Nppp_right_interface)
            # penalty term contribution at the current interface to the system stiffness
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += (1.0-self.internal_variables[i:i+1, 1:2])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), N_left_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), Np_left_interface)))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (1.0-self.internal_variables[i:i+1, 1:2])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), N_right_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), Np_right_interface)))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (1.0-self.internal_variables[i:i+1, 1:2])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), N_right_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), Np_right_interface)))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= (1.0-self.internal_variables[i:i+1, 1:2])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), N_left_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), Np_left_interface)))
            # adding 't_i' vector gradient contributions at the current interface to the system stiffness
            # works only when there are no elemental loads!!!
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= 0.50*(1.0-self.internal_variables[i:i+1, 1:2])*((self.material.E*self.material.A*np.matmul(np.transpose(N_left_interface), dt1dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_left_interface), dt5dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_left_interface), dt3dd_left_interface)))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += 0.50*(1.0-self.internal_variables[i:i+1, 1:2])*((self.material.E*self.material.A*np.matmul(np.transpose(N_right_interface), dt1dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_right_interface), dt5dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_right_interface), dt3dd_left_interface)))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= 0.50*(1.0-self.internal_variables[i:i+1, 1:2])*((self.material.E*self.material.A*np.matmul(np.transpose(N_left_interface), dt1dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_left_interface), dt5dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_left_interface), dt3dd_right_interface)))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += 0.50*(1.0-self.internal_variables[i:i+1, 1:2])*((self.material.E*self.material.A*np.matmul(np.transpose(N_right_interface), dt1dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_right_interface), dt5dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_right_interface), dt3dd_right_interface)))
