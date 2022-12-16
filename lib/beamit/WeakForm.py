import numpy as np
from beamit import Material

def cross_op(arr1:np.ndarray,arr2:np.ndarray,a:int,b:int,c:int)->np.ndarray:
    return np.cross(arr1,arr2,axisa=a,axisb=b,axisc=c)

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
        mdist_cross_t4 = cross_op(el_dist_moments, t4, 1, 0, 1)
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
            f[global_element_dofs_right_node] -= self.compute_element_internal_forces(element_unknowns)[dofs:dofspel]
            # correcting the sign of axial forces
            f[np.array([global_element_dofs_right_node[0]])] *= -1.0
            # Assuming the elements are connected like a simple chain!!!
            if (i == 0): # only for the first element
                global_element_dofs_left_node = global_element_dofs[0:dofs]
                f[global_element_dofs_left_node] += self.compute_element_internal_forces(element_unknowns)[0:dofs]
                # correcting the sign of axial forces
                f[np.array([global_element_dofs_left_node[0]])] *= -1.0
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

    def __init__(self, function_space, material, betaP, betaT):
        # invoke the parent (WeakFormCG) class
        WeakFormCG.__init__(self, function_space, material)
        # DG position jump penalty parameter
        self.betaP = betaP
        # DG tangent jump penalty parameter
        self.betaT = betaT
        # store internal variables (of CZM) at all the interfaces
        # first column = binary operator to indicate damage status (0.0 -> damage 'not' initiated, 1.0 -> damage initiated) at the interface
        # second column = maximum effective separation at an interface in the entire loading history
        self.internal_variables = np.zeros([self.function_space.E-1, 2])
    
    # Helper function to compute 't_i' vectors in the residual
    def compute_residual_vectors(self, rp, rpp, rppp):
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
    
    # Function to compute the system residual
    def compute_system_residual(self, f, system_unknowns, element_loads_info):
        # compute system residual using the function in WeakFormCG
        super().compute_system_residual(f, system_unknowns, element_loads_info)
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
            t1_left_interface, _, _, t4_left_interface, t5_left_interface = self.compute_residual_vectors(rp_left_interface, rpp_left_interface, rppp_left_interface)
            t1_right_interface, _, _, t4_right_interface, t5_right_interface = self.compute_residual_vectors(rp_right_interface, rpp_right_interface, rppp_right_interface)
            # position and tangent jumps at the interface
            r_jump_interface = r_right_interface - r_left_interface
            rp_jump_interface = rp_right_interface - rp_left_interface
            # forces at the interface
            if (element_loads_info == None): # No element loads
                forces_left_interface = (self.material.E*self.material.A*t1_left_interface) + (self.material.E*self.material.I*t5_left_interface)
                forces_right_interface = (self.material.E*self.material.A*t1_right_interface) + (self.material.E*self.material.I*t5_right_interface)
            average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
            # moments at the interface
            moments_left_interface = self.material.E*self.material.I*(cross_op(rp_left_interface, rpp_left_interface, 0, 0, 0)/(rp_left_interface_L2**2.0))
            moments_right_interface = self.material.E*self.material.I*(cross_op(rp_right_interface, rpp_right_interface, 0, 0, 0)/(rp_right_interface_L2**2.0))
            mxt4_left_interface = cross_op(moments_left_interface, t4_left_interface, 0, 0, 0)
            mxt4_right_interface = cross_op(moments_right_interface, t4_right_interface, 0, 0, 0)
            average_mxt4_interface = (mxt4_left_interface + mxt4_right_interface)/2.0
            # initialize interface forces
            interface_forces = np.zeros(average_forces_interface.shape)
            # perform CZM checks and calculations in the case of a cohesive interface material
            if (isinstance(self.material, (Material.CohesiveInterfaceMaterial))):
                # damage not yet initiated at the interface
                if ((self.internal_variables[i:i+1, 0:1] == 0.0) and (self.internal_variables[i:i+1, 1:2] == 0.0)):
                    # evaluate the damage initiation criterion
                    if (self.material.evaluate_damage_initiation_criterion(rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface)):
                        # damage just initiated at the interface
                        self.internal_variables[i:i+1, 0:1] = 1.0
                        # evaluate interface forces according to the TSL
                        interface_forces = self.material.compute_cohesive_forces(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, self.internal_variables[i:i+1, 1:2])
                    # else - No damage at the interface (do nothing)!!!
                # elif - damage after recontact at the interface!!!
                else: # damage already initiated at the interface
                    # if - recontact after damage at the interface!!!
                    # evaluate interface forces according to the TSL
                    interface_forces = self.material.compute_cohesive_forces(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, self.internal_variables[i:i+1, 1:2])
            # adding the contributions of the current interface to the system residual
            # DG FLUX AND COMPATIBILITY TERMS
            f[global_element_dofs_left] += (((1.0-self.internal_variables[i:i+1, 0:1])*np.matmul(np.transpose(N_left_interface), average_forces_interface)) + np.matmul(np.transpose(Np_left_interface), average_mxt4_interface) + ((1.0-self.internal_variables[i:i+1, 0:1])*self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), r_jump_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), rp_jump_interface)))
            f[global_element_dofs_right] -= (((1.0-self.internal_variables[i:i+1, 0:1])*np.matmul(np.transpose(N_right_interface), average_forces_interface)) + np.matmul(np.transpose(Np_right_interface), average_mxt4_interface) + ((1.0-self.internal_variables[i:i+1, 0:1])*self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), r_jump_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), rp_jump_interface)))
            # INTERFACE FORCES FROM THE CZM
            f[global_element_dofs_left] += (self.internal_variables[i:i+1, 0:1]*(np.matmul(np.transpose(N_left_interface), interface_forces)))
            f[global_element_dofs_right] -= (self.internal_variables[i:i+1, 0:1]*(np.matmul(np.transpose(N_right_interface), interface_forces)))
        pass
    
    # Helper function to compute coefficients of 't_i' vector gradients in the jump stiffness
    def compute_jump_stiffness_coefficients(self, rp, rpp, rppp):
        imat = np.eye(self.function_space.dim)
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
    def compute_system_stiffness(self, A, system_unknowns):
        # compute system stiffness using the function in WeakFormCG
        super().compute_system_stiffness(A, system_unknowns)
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
            r_left_interface = np.matmul(N_left_interface, element_unknowns_left)
            r_right_interface = np.matmul(N_right_interface, element_unknowns_right)
            rp_left_interface = np.matmul(Np_left_interface, element_unknowns_left)
            rp_right_interface = np.matmul(Np_right_interface, element_unknowns_right)
            rpp_left_interface = np.matmul(Npp_left_interface, element_unknowns_left)
            rpp_right_interface = np.matmul(Npp_right_interface, element_unknowns_right)
            rppp_left_interface = np.matmul(Nppp_left_interface, element_unknowns_left)
            rppp_right_interface = np.matmul(Nppp_right_interface, element_unknowns_right)
            # FLUX AND COMPATIBILITY TERM DERIVATIVES
            # coefficients of 't_i' vector gradients at the interface
            dt1dd_Np_left_interface, dt3dd_Np_left_interface, dt3dd_Npp_left_interface, dt5dd_Np_left_interface, dt5dd_Npp_left_interface, dt5dd_Nppp_left_interface = self.compute_jump_stiffness_coefficients(rp_left_interface, rpp_left_interface, rppp_left_interface)
            dt1dd_Np_right_interface, dt3dd_Np_right_interface, dt3dd_Npp_right_interface, dt5dd_Np_right_interface, dt5dd_Npp_right_interface, dt5dd_Nppp_right_interface = self.compute_jump_stiffness_coefficients(rp_right_interface, rpp_right_interface, rppp_right_interface)
            # 't_i' vector gradients at the interface
            dt1dd_left_interface = np.matmul(dt1dd_Np_left_interface, Np_left_interface)
            dt1dd_right_interface = np.matmul(dt1dd_Np_right_interface, Np_right_interface)
            dt3dd_left_interface = np.matmul(dt3dd_Np_left_interface, Np_left_interface) + np.matmul(dt3dd_Npp_left_interface, Npp_left_interface)
            dt3dd_right_interface = np.matmul(dt3dd_Np_right_interface, Np_right_interface) + np.matmul(dt3dd_Npp_right_interface, Npp_right_interface)
            dt5dd_left_interface = np.matmul(dt5dd_Np_left_interface, Np_left_interface) + np.matmul(dt5dd_Npp_left_interface, Npp_left_interface) + np.matmul(dt5dd_Nppp_left_interface, Nppp_left_interface)
            dt5dd_right_interface = np.matmul(dt5dd_Np_right_interface, Np_right_interface) + np.matmul(dt5dd_Npp_right_interface, Npp_right_interface) + np.matmul(dt5dd_Nppp_right_interface, Nppp_right_interface)
            # initialize the force derivative coefficients at the interface
            dft1dd_N_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dft1dd_Np_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dft2dd_Np_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            # perform CZM checks and calculations in the case of a cohesive interface material
            if (isinstance(self.material, (Material.CohesiveInterfaceMaterial))):
                # only if the interface is already damaged
                if (self.internal_variables[i:i+1, 0:1] == 1.0):
                    dft1dd_N_coeff, dft1dd_Np_coeff, dft2dd_Np_coeff = self.material.compute_cohesive_force_derivative_coefficients(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, self.internal_variables[i:i+1, 1:2])
                    # effective separation at the interface
                    delta = self.material.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface)
                    # update the maximum effective separation
                    new_delta_max = self.material.compute_effective_maximum_separation(delta, self.internal_variables[i:i+1, 1:2])
                    self.internal_variables[i:i+1, 1:2] = new_delta_max
            # penalty term contribution at the current interface to the system stiffness
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += ((1.0-self.internal_variables[i:i+1, 0:1])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), N_left_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), Np_left_interface))))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += ((1.0-self.internal_variables[i:i+1, 0:1])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), N_right_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), Np_right_interface))))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= ((1.0-self.internal_variables[i:i+1, 0:1])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), N_right_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), Np_right_interface))))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= ((1.0-self.internal_variables[i:i+1, 0:1])*((self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), N_left_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), Np_left_interface))))
            # adding 't_i' vector gradient contributions at the current interface to the system stiffness
            # works only when there are no elemental loads!!!
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= ((1.0-self.internal_variables[i:i+1, 0:1])*0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_left_interface), dt1dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_left_interface), dt5dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_left_interface), dt3dd_left_interface))))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += ((1.0-self.internal_variables[i:i+1, 0:1])*0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_right_interface), dt1dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_right_interface), dt5dd_left_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_right_interface), dt3dd_left_interface))))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= ((1.0-self.internal_variables[i:i+1, 0:1])*0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_left_interface), dt1dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_left_interface), dt5dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_left_interface), dt3dd_right_interface))))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += ((1.0-self.internal_variables[i:i+1, 0:1])*0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_right_interface), dt1dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(N_right_interface), dt5dd_right_interface)) + (self.material.E*self.material.I*np.matmul(np.transpose(Np_right_interface), dt3dd_right_interface))))
            # INTERFACE FORCE DERIVATIVES FROM THE CZM
            dft1dd_term1_left = np.matmul(dft1dd_N_coeff, N_left_interface)
            dft1dd_term1_right = np.matmul(dft1dd_N_coeff, N_right_interface)
            dft1dd_term2_left = np.matmul(dft1dd_Np_coeff, Np_left_interface)
            dft1dd_term2_right = np.matmul(dft1dd_Np_coeff, Np_right_interface)
            dft2dd_left = np.matmul(dft2dd_Np_coeff, Np_left_interface)
            dft2dd_right = np.matmul(dft2dd_Np_coeff, Np_right_interface)
            # first set of terms
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += (self.internal_variables[i:i+1, 0:1]*np.matmul(np.transpose(N_left_interface), dft1dd_term1_left))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.internal_variables[i:i+1, 0:1]*np.matmul(np.transpose(N_right_interface), dft1dd_term1_right))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.internal_variables[i:i+1, 0:1]*np.matmul(np.transpose(N_left_interface), dft1dd_term1_right))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= (self.internal_variables[i:i+1, 0:1]*np.matmul(np.transpose(N_right_interface), dft1dd_term1_left))
            # second set of terms
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= (self.internal_variables[i:i+1, 0:1]*(np.matmul(np.transpose(N_left_interface), dft1dd_term2_left) + np.matmul(np.transpose(N_left_interface), dft2dd_left)))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += (self.internal_variables[i:i+1, 0:1]*(np.matmul(np.transpose(N_right_interface), dft1dd_term2_left) + np.matmul(np.transpose(N_right_interface), dft2dd_left)))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.internal_variables[i:i+1, 0:1]*(np.matmul(np.transpose(N_left_interface), dft1dd_term2_right) + np.matmul(np.transpose(N_left_interface), dft2dd_right)))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.internal_variables[i:i+1, 0:1]*(np.matmul(np.transpose(N_right_interface), dft1dd_term2_right) + np.matmul(np.transpose(N_right_interface), dft2dd_right)))
        pass

    # Function to compute the system internal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_internal_forces(self, f, system_unknowns):
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_left_node = global_element_dofs[0:dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            f[global_element_dofs_left_node] += self.compute_element_internal_forces(element_unknowns)[0:dofs]
            f[global_element_dofs_right_node] -= self.compute_element_internal_forces(element_unknowns)[dofs:dofspel]
            # correcting the sign of axial forces
            f[np.array([global_element_dofs_left_node[0]])] *= -1.0
            f[np.array([global_element_dofs_right_node[0]])] *= -1.0
        pass
