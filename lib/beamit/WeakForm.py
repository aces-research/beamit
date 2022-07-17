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
        # A SIGN PROBLEM HERE!!!
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

    # Function to compute the system stiffness
    def compute_system_stiffness(self, A, system_unknowns):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            A[np.ix_(global_element_dofs, global_element_dofs)] += self.compute_element_internal_stiffness(element_unknowns)
        pass