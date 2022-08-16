import numpy as np

class WeakFormCG:
    
    def __init__(self, function_space, material):
        # the geometrical information
        self.function_space = function_space
        # the physical information
        self.material = material

    # Function to compute element internal forces
    def compute_element_internal_forces(self, element_unknowns):
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
        dt4dd = np.matmul(((1.0/(rp_L2**2.0))*imat - (2.0/(rp_L2**4.0))*(rp_dyd_rp)), Np)
        axial_integrand = self.material.E*self.material.A*np.matmul(np.transpose(Np, axes=(0, 2, 1)), dt1dd)
        bending_integrand_1 = self.material.E*self.material.I*np.matmul(np.transpose(Np, axes=(0, 2, 1)), dt2dd)
        bending_integrand_2 = self.material.E*self.material.I*np.matmul(np.transpose(Npp, axes=(0, 2, 1)), dt3dd)
        K_el_int = np.sum((axial_integrand + bending_integrand_1 + bending_integrand_2)*self.function_space.JxW, axis=0, keepdims=False)
        return K_el_int

    # Function to compute element external distributed forces (for distributed forces and moments)
    def compute_element_external_distributed_forces(self, element_unknowns, el_dist_forces, el_dist_moments):
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

    # Function to compute the overall system residual
    def compute_system_residual(self, f, system_unknowns, element_loads_info):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads_info == None): # No element loads
                f[global_element_dofs] -= self.compute_element_internal_forces(element_unknowns)
        pass

    # Function to compute the system internal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_internal_forces(self, f, system_unknowns):
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            f[global_element_dofs_right_node] += self.compute_element_internal_forces(element_unknowns)[dofs:dofspel]
            # Assuming the elements are connected like a simple chain!!!
            if (i == 0): # only for the first element
                global_element_dofs_left_node = global_element_dofs[0:dofs]
                f[global_element_dofs_left_node] -= self.compute_element_internal_forces(element_unknowns)[0:dofs]
        pass

    # Function to compute the system stiffness
    def compute_system_stiffness(self, A, system_unknowns):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            A[np.ix_(global_element_dofs, global_element_dofs)] += self.compute_element_internal_stiffness(element_unknowns)
        pass
    
    # Function to compute the system mass
    def compute_system_mass(self, M):
        Nt = np.transpose(self.function_space.shape_functions, axes=(0, 2, 1))
        mass_integrand = self.material.rho*self.material.A*np.matmul(Nt, self.function_space.shape_functions)
        # the element mass matrix
        M_el = np.sum(mass_integrand*self.function_space.JxW, axis=0, keepdims=False)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            M[np.ix_(global_element_dofs, global_element_dofs)] += M_el
        pass

class WeakFormDG(WeakFormCG):

    def __init__(self, function_space, material):
        # invoke the parent (WeakFormCG) class
        WeakFormCG.__init__(self, function_space, material)
        # the DG penalty parameter
        self.beta = 10.0
    
    # Function to compute the overall system residual
    def compute_system_residual(self, f, system_unknowns, element_loads_info):
        # compute system residual using the function in WeakFormCG
        super().compute_system_residual(f, system_unknowns, element_loads_info)
        # add the contributions of the jumps at the interfaces to the residual
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        for i in range(0, self.function_space.E-1):
            # since the elements are placed one after the other like a simple chain!!!
            # current element (= left (-)) and next element (= right (+))
            global_element_dofs_left = self.function_space.global_connectivity[i:i+1].flatten()
            global_element_dofs_right = self.function_space.global_connectivity[i+1:i+2].flatten()
            # unknowns of the left and right elements
            element_unknowns_left = system_unknowns[global_element_dofs_left]
            element_unknowns_right = system_unknowns[global_element_dofs_right]
            # unknowns of the left and right elements at the interface
            element_unknowns_left_interface = element_unknowns_left[dofs:dofspel, :]
            element_unknowns_right_interface = element_unknowns_right[0:dofs, :]
            # internal forces of the left and right elements at the interface
            internal_forces_left_interface = self.compute_element_internal_forces(element_unknowns_left)[dofs:dofspel, :]
            internal_forces_right_interface = self.compute_element_internal_forces(element_unknowns_right)[0:dofs, :]
            # global dof numbering of the current interface
            interface_global_dofs = np.concatenate([global_element_dofs_left[dofs:dofspel], global_element_dofs_right[0:dofs]])
            # internal "loads" (= f) and moments (= m) at the interface - CHECK!!!
            internal_loads_left_interface = internal_forces_left_interface[0:dofs/2, :]
            internal_loads_right_interface = internal_forces_right_interface[0:dofs/2, :]
            internal_moments_left_interface = internal_forces_left_interface[dofs/2:dofs, :]
            internal_moments_right_interface = internal_forces_right_interface[dofs/2:dofs, :]
            # computing the average internal "loads" (= <f>) at the interface
            average_loads_interface = (internal_loads_left_interface + internal_loads_right_interface)/2.0
            # positions and tangents at the interface
            positions_left_interface = element_unknowns_left_interface[0:dofs/2, :]
            positions_right_interface = element_unknowns_right_interface[0:dofs/2, :]
            tangents_left_interface = element_unknowns_left_interface[dofs/2:dofs, :]
            tangents_right_interface = element_unknowns_right_interface[dofs/2:dofs, :]
            # unit tangents and 't4's at the interface
            tangents_left_interface_L2 = np.linalg.norm(tangents_left_interface, ord=2, axis=0, keepdims=True)
            tangents_right_interface_L2 = np.linalg.norm(tangents_right_interface, ord=2, axis=0, keepdims=True)
            t4_left_interface = tangents_left_interface/(tangents_left_interface_L2**2.0)
            t4_right_interface = tangents_right_interface/(tangents_right_interface_L2**2.0)
            # computing the average internal moments (= <m x t_4>) at the interface
            mxt4_left_interface = np.cross(internal_moments_left_interface, t4_left_interface, axisa=0, axisb=0, axisc=0)
            mxt4_right_interface = np.cross(internal_moments_right_interface, t4_right_interface, axisa=0, axisb=0, axisc=0)
            average_moments_interface = (mxt4_left_interface + mxt4_right_interface)/2.0