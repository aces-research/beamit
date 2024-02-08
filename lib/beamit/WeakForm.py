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
    
    # Function to compute element inertia forces
    def compute_element_inertia_forces(self, element_unknowns, element_velocities, element_accelerations):
        Nt = np.transpose(self.function_space.shape_functions, axes=(0, 2, 1))
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npt = np.transpose(Np, axes=(0, 2, 1))
        rp = np.matmul(Np, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dot = np.matmul(Np, element_velocities)
        rp_dot_rp_dot = np.sum(rp*rp_dot, axis=1, keepdims=True)
        r_ddot = np.matmul(self.function_space.shape_functions, element_accelerations)
        rp_ddot = np.matmul(Np, element_accelerations)
        rp_dot_rp_ddot = np.sum(rp*rp_ddot, axis=1, keepdims=True)
        translational_integrand = self.material.rho*self.material.A*np.matmul(Nt, r_ddot)
        rotational_integrand_1 = ((self.material.rho*self.material.I)/(rp_L2**2.0))*np.matmul(Npt, rp_ddot)
        rotational_integrand_2 = ((2.0*self.material.rho*self.material.I*(rp_dot_rp_dot**2.0))/(rp_L2**6.0))*np.matmul(Npt, rp)
        rotational_integrand_3 = -((2.0*self.material.rho*self.material.I*rp_dot_rp_dot)/(rp_L2**4.0))*np.matmul(Npt, rp_dot)
        rotational_integrand_4 = -((self.material.rho*self.material.I*rp_dot_rp_ddot)/(rp_L2**4.0))*np.matmul(Npt, rp)
        f_el_inertia = np.sum((translational_integrand + rotational_integrand_1 + rotational_integrand_2 + rotational_integrand_3 + rotational_integrand_4)*self.function_space.JxW, axis=0, keepdims=False)
        return f_el_inertia

    # Function to compute the element "damping inertia forces"
    def compute_element_damping_inertia_forces(self, element_unknowns, element_velocities):
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npt = np.transpose(Np, axes=(0, 2, 1))
        rp = np.matmul(Np, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dot = np.matmul(Np, element_velocities)
        rp_dot_rp_dot = np.sum(rp*rp_dot, axis=1, keepdims=True)
        damping_integrand_1 = ((2.0*self.material.rho*self.material.I*(rp_dot_rp_dot**2.0))/(rp_L2**6.0))*np.matmul(Npt, rp)
        damping_integrand_2 = -((2.0*self.material.rho*self.material.I*rp_dot_rp_dot)/(rp_L2**4.0))*np.matmul(Npt, rp_dot)
        f_el_damping = np.sum((damping_integrand_1 + damping_integrand_2)*self.function_space.JxW, axis=0, keepdims=False)
        return f_el_damping
    
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

    # Function to compute element rotational mass
    def compute_element_rotational_mass(self, element_unknowns):
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npt = np.transpose(Np, axes=(0, 2, 1))
        rp = np.matmul(Np, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dyd_rp = np.matmul(rp, np.transpose(rp, axes=(0, 2, 1)))
        rotational_mass_integrand_1 = ((self.material.rho*self.material.I)/(rp_L2**2.0))*np.matmul(Npt, Np)
        rotational_mass_integrand_2 = -((self.material.rho*self.material.I)/(rp_L2**4.0))*np.matmul(Npt, np.matmul(rp_dyd_rp, Np))
        M_el_rot = np.sum((rotational_mass_integrand_1 + rotational_mass_integrand_2)*self.function_space.JxW, axis=0, keepdims=False)
        return M_el_rot

    # Function to compute element damping
    def compute_element_damping(self, element_unknowns, element_velocities):
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npt = np.transpose(Np, axes=(0, 2, 1))
        rp = np.matmul(Np, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dyd_rp = np.matmul(rp, np.transpose(rp, axes=(0, 2, 1)))
        rp_dot = np.matmul(Np, element_velocities)
        rp_dot_rp_dot = np.sum(rp*rp_dot, axis=1, keepdims=True)
        rp_dot_dyd_rp = np.matmul(rp_dot, np.transpose(rp, axes=(0, 2, 1)))
        damping_integrand_1 = ((4.0*self.material.rho*self.material.I*rp_dot_rp_dot)/(rp_L2**6.0))*np.matmul(Npt, np.matmul(rp_dyd_rp, Np))
        damping_integrand_2 = -((2.0*self.material.rho*self.material.I*rp_dot_rp_dot)/(rp_L2**4.0))*np.matmul(Npt, Np)
        damping_integrand_3 = -((2.0*self.material.rho*self.material.I)/(rp_L2**4.0))*np.matmul(Npt, np.matmul(rp_dot_dyd_rp, Np))
        M_el_damp = np.sum((damping_integrand_1 + damping_integrand_2 + damping_integrand_3)*self.function_space.JxW, axis=0, keepdims=False)
        return M_el_damp

    # Function to compute the element rotational inertia stiffness
    def compute_element_rotational_inertia_stiffness(self, element_unknowns, element_velocities, element_accelerations):
        Np = self.function_space.shape_first_gradients*(1.0/self.function_space.jacobian)
        Npt = np.transpose(Np, axes=(0, 2, 1))
        rp = np.matmul(Np, element_unknowns)
        rp_L2 = np.linalg.norm(rp, ord=2, axis=1, keepdims=True)
        rp_dyd_rp = np.matmul(rp, np.transpose(rp, axes=(0, 2, 1)))
        rp_dot = np.matmul(Np, element_velocities)
        rp_dot_rp_dot = np.sum(rp*rp_dot, axis=1, keepdims=True)
        rp_ddot = np.matmul(Np, element_accelerations)
        rp_dot_rp_ddot = np.sum(rp*rp_ddot, axis=1, keepdims=True)
        rp_dyd_rp_dot = np.matmul(rp, np.transpose(rp_dot, axes=(0, 2, 1)))
        rp_dot_dyd_rp = np.matmul(rp_dot, np.transpose(rp, axes=(0, 2, 1)))
        rp_dot_dyd_rp_dot = np.matmul(rp_dot, np.transpose(rp_dot, axes=(0, 2, 1)))
        rp_dyd_rp_ddot = np.matmul(rp, np.transpose(rp_ddot, axes=(0, 2, 1)))
        rp_ddot_dyd_rp = np.matmul(rp_ddot, np.transpose(rp, axes=(0, 2, 1)))
        rot_stiff_integrand_1 = -((2.0*self.material.rho*self.material.I)/(rp_L2**4.0))*np.matmul(Npt, np.matmul(rp_ddot_dyd_rp, Np))
        rot_stiff_integrand_2 = (((8.0*self.material.rho*self.material.I*rp_dot_rp_dot)/(rp_L2**6.0))*np.matmul(Npt, np.matmul(rp_dot_dyd_rp, Np))) -(((2.0*self.material.rho*self.material.I)/(rp_L2**4.0))*np.matmul(Npt, np.matmul(rp_dot_dyd_rp_dot, Np)))
        rot_stiff_integrand_3 = (((2.0*self.material.rho*self.material.I*(rp_dot_rp_dot**2.0))/(rp_L2**6.0))*np.matmul(Npt, Np)) + (((4.0*self.material.rho*self.material.I*rp_dot_rp_dot)/(rp_L2**6.0))*np.matmul(Npt, np.matmul(rp_dyd_rp_dot, Np))) - (((12.0*self.material.rho*self.material.I*(rp_dot_rp_dot**2.0))/(rp_L2**8.0))*np.matmul(Npt, np.matmul(rp_dyd_rp, Np)))
        rot_stiff_integrand_4 = (((4.0*self.material.rho*self.material.I*rp_dot_rp_ddot)/(rp_L2**6.0))*np.matmul(Npt, np.matmul(rp_dyd_rp, Np))) - (((self.material.rho*self.material.I*rp_dot_rp_ddot)/(rp_L2**4.0))*np.matmul(Npt, Np)) - (((self.material.rho*self.material.I)/(rp_L2**4.0))*np.matmul(Npt, np.matmul(rp_dyd_rp_ddot, Np)))
        K_el_rot_inertia = np.sum((rot_stiff_integrand_1 + rot_stiff_integrand_2 + rot_stiff_integrand_3 + rot_stiff_integrand_4)*self.function_space.JxW, axis=0, keepdims=False)
        return K_el_rot_inertia

    # Function to compute the overall system residual
    def compute_system_residual(self, f, system_unknowns, element_loads_info):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads_info == None): # No element loads
                f[global_element_dofs] -= self.compute_element_internal_forces(element_unknowns)
        pass

    # Function to compute the system "damping inertia forces" contribution to the residual
    def compute_system_damping_inertia_forces(self, f, system_unknowns, system_velocities):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            element_velocities = system_velocities[global_element_dofs]
            f[global_element_dofs] -= self.compute_element_damping_inertia_forces(element_unknowns, element_velocities)
        pass
    
    # Function to compute the system inertia forces contribution to the residual
    def compute_system_inertia_forces(self, f, system_unknowns, system_velocities, system_accelerations):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            element_velocities = system_velocities[global_element_dofs]
            element_accelerations = system_accelerations[global_element_dofs]
            f[global_element_dofs] -= self.compute_element_inertia_forces(element_unknowns, element_velocities, element_accelerations)
        pass
    
    # Helper function to compute 't_i' vectors
    def compute_ti_vectors(self, rp, rpp, rppp):
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
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads_info=None):
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
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            rp_right_node = np.matmul(Np_right_node, element_unknowns)
            rpp_right_node = np.matmul(Npp_right_node, element_unknowns)
            rppp_right_node = np.matmul(Nppp_right_node, element_unknowns)
            rp_right_node_L2 = np.linalg.norm(rp_right_node, ord=2, axis=0, keepdims=True) 
            t1_right_node, _, _, _, t5_right_node = self.compute_ti_vectors(rp_right_node, rpp_right_node, rppp_right_node)
            # forces at the right node
            if (element_loads_info == None): # No element loads
                forces_right_node = (self.material.E*self.material.A*t1_right_node) + (self.material.E*self.material.I*t5_right_node)
            # moments at the right node
            moments_right_node = self.material.E*self.material.I*(cross_op(rp_right_node, rpp_right_node, 0, 0, 0)/(rp_right_node_L2**2.0))
            f[global_element_dofs_right_node[0:int(dofs/2)]] += forces_right_node
            f[global_element_dofs_right_node[int(dofs/2):dofs]] += moments_right_node
            # Assuming the elements are connected like a simple chain!!!
            if (i == 0): # only for the first element
                global_element_dofs_left_node = global_element_dofs[0:dofs]
                rp_left_node = np.matmul(Np_left_node, element_unknowns)
                rpp_left_node = np.matmul(Npp_left_node, element_unknowns)
                rppp_left_node = np.matmul(Nppp_left_node, element_unknowns)
                rp_left_node_L2 = np.linalg.norm(rp_left_node, ord=2, axis=0, keepdims=True)
                t1_left_node, _, _, _, t5_left_node = self.compute_ti_vectors(rp_left_node, rpp_left_node, rppp_left_node)
                # forces at the left node
                if (element_loads_info == None): # No element loads
                    forces_left_node = (self.material.E*self.material.A*t1_left_node) + (self.material.E*self.material.I*t5_left_node)
                # moments at the left node
                moments_left_node = self.material.E*self.material.I*(cross_op(rp_left_node, rpp_left_node, 0, 0, 0)/(rp_left_node_L2**2.0))
                f[global_element_dofs_left_node[0:int(dofs/2)]] += forces_left_node
                f[global_element_dofs_left_node[int(dofs/2):dofs]] += moments_left_node
        pass

    # Function to compute the system stiffness
    def compute_system_stiffness(self, A, system_unknowns, element_loads_info):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads_info == None): # No element loads
                A[np.ix_(global_element_dofs, global_element_dofs)] += self.compute_element_internal_stiffness(element_unknowns)
        pass
    
    # Function to compute the system mass
    def compute_system_mass(self, M, system_unknowns, use_rotational_mass=False, lump=True):
        Nt = np.transpose(self.function_space.shape_functions, axes=(0, 2, 1))
        translational_mass_integrand = self.material.rho*self.material.A*np.matmul(Nt, self.function_space.shape_functions)
        # the element translational mass matrix
        M_el_trans = np.sum(translational_mass_integrand*self.function_space.JxW, axis=0, keepdims=False)
        # apply "special lumping" to the element translational mass matrix
        if (lump):
            sum_all_entries = np.sum(M_el_trans)
            sum_diag_entries = np.sum(np.diag(M_el_trans))
            diag_elements = (sum_all_entries / sum_diag_entries) * np.diag(M_el_trans)
            M_el_trans = np.zeros([self.function_space.npel*self.function_space.dof, self.function_space.npel*self.function_space.dof])
            np.fill_diagonal(M_el_trans, diag_elements)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            M[np.ix_(global_element_dofs, global_element_dofs)] += M_el_trans
            if (use_rotational_mass):
                M_el_rot = self.compute_element_rotational_mass(element_unknowns)
                # apply "special lumping" to the element rotational mass matrix
                if (lump):
                    sum_all_entries = np.sum(M_el_rot)
                    sum_diag_entries = np.sum(np.diag(M_el_rot))
                    diag_elements = (sum_all_entries / sum_diag_entries) * np.diag(M_el_rot)
                    M_el_rot = np.zeros([self.function_space.npel*self.function_space.dof, self.function_space.npel*self.function_space.dof])
                    np.fill_diagonal(M_el_rot, diag_elements)
                M[np.ix_(global_element_dofs, global_element_dofs)] += M_el_rot
        pass

    # Function to compute the system damping
    def compute_system_damping(self, C, system_unknowns, system_velocities):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            element_velocities = system_velocities[global_element_dofs]
            C[np.ix_(global_element_dofs, global_element_dofs)] += self.compute_element_damping(element_unknowns, element_velocities)
        pass

    # Function to compute the system rotational inertia stiffness
    def compute_system_rotational_inertia_stiffness(self, A, system_unknowns, system_velocities, system_accelerations):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            element_velocities = system_velocities[global_element_dofs]
            element_accelerations = system_accelerations[global_element_dofs]
            A[np.ix_(global_element_dofs, global_element_dofs)] += self.compute_element_rotational_inertia_stiffness(element_unknowns, element_velocities, element_accelerations)
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
    def compute_system_residual(self, f, system_unknowns, element_loads_info, update_internal=False):
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
            # initialize cohesive forces and bending moments
            cohesive_forces = np.zeros(average_forces_interface.shape)
            cohesive_bending_moments = np.zeros(average_mxt4_interface.shape)
            # perform CZM checks and calculations in the case of a cohesive interface material
            if (isinstance(self.material, (Material.CohesiveInterfaceMaterial))):
                # just after damage initiation or damage not yet initiated
                if (self.internal_variables[i:i+1, 2:3] == 0.0):
                    # just after damage initiation
                    if (self.internal_variables[i:i+1, 0:1] == 1.0):
                        # effective separation at the interface
                        delta = self.material.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                        if (delta == 0.0): # fall back to DG terms
                            self.internal_variables[i:i+1, 1:2] = 0.0
                        else: # perform CZM calculations
                            self.internal_variables[i:i+1, 1:2] = 1.0
                            axial_jump = self.material.compute_axial_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=self.internal_variables[i:i+1, 4:7].T)
                            if (axial_jump < 0.0): # recontact at the interface - NOT USING THIS ONE FOR NOW!!!
                                self.internal_variables[i:i+1, 10:11] = 0.0
                            else:
                                self.internal_variables[i:i+1, 10:11] = 1.0
                            # evaluate cohesive forces according to the TSL
                            cohesive_axial_forces = self.material.compute_cohesive_axial_forces(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max=self.internal_variables[i:i+1, 2:3], position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                            # compute the cohesive forces and bending moments
                            if ((delta >= self.material.delta_c) or (self.internal_variables[i:i+1, 2:3] == self.material.delta_c)): # for complete damage
                                interface_constrained_shear_forces = np.zeros(average_forces_interface.shape)
                            else:
                                effective_unit_tangent_interface = self.material.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
                                tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
                                interface_constrained_shear_forces = np.matmul((np.eye(3)-tangeff_dyd_tangeff), average_forces_interface) + (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul((np.eye(3)-tangeff_dyd_tangeff), r_jump_interface))
                            cohesive_forces = cohesive_axial_forces + interface_constrained_shear_forces
                            cohesive_bending_moments = self.material.compute_cohesive_bending_moments(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max=self.internal_variables[i:i+1, 2:3], position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                            if (update_internal):
                                # update the maximum effective separation
                                new_delta_max = self.material.compute_effective_maximum_separation(delta, delta_max=self.internal_variables[i:i+1, 2:3])
                                self.internal_variables[i:i+1, 2:3] = new_delta_max
                    # damage not yet initiated
                    elif (self.material.evaluate_damage_initiation_criterion(rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface)): # evaluate the damage initiation criterion
                        # damage just initiated at the interface
                        self.internal_variables[i:i+1, 0:1] = 1.0
                        # keep the DG flux and compatibility terms active immediately after damage initiation (since delta = 0.0)
                        self.internal_variables[i:i+1, 1:2] = 0.0
                        # update the internal variables at damage initiation
                        self.internal_variables[i:i+1, 3:4] = self.material.compute_effective_force(rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface)
                        self.internal_variables[i:i+1, 4:7] = r_jump_interface.T
                        self.internal_variables[i:i+1, 7:10] = rp_jump_interface.T
                    else: # no damage at the interface
                        self.internal_variables[i:i+1, 1:2] = 0.0
                else: # damage already initiated at the interface (loading | unloading | damage after recontact)
                    # effective separation at the interface
                    delta = self.material.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                    if (delta == 0.0): # fall back to DG terms
                        self.internal_variables[i:i+1, 1:2] = 0.0
                    else: # perform CZM calculations
                        self.internal_variables[i:i+1, 1:2] = 1.0
                        axial_jump = self.material.compute_axial_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=self.internal_variables[i:i+1, 4:7].T)
                        if (axial_jump < 0.0): # recontact at the interface - NOT USING THIS ONE FOR NOW!!!
                            self.internal_variables[i:i+1, 10:11] = 0.0
                        else:
                            self.internal_variables[i:i+1, 10:11] = 1.0
                        # evaluate cohesive forces according to the TSL
                        cohesive_axial_forces = self.material.compute_cohesive_axial_forces(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max=self.internal_variables[i:i+1, 2:3], position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                        # compute the cohesive forces and bending moments
                        if ((delta >= self.material.delta_c) or (self.internal_variables[i:i+1, 2:3] == self.material.delta_c)): # for complete damage
                            interface_constrained_shear_forces = np.zeros(average_forces_interface.shape)
                        else:
                            effective_unit_tangent_interface = self.material.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
                            tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
                            interface_constrained_shear_forces = np.matmul((np.eye(3)-tangeff_dyd_tangeff), average_forces_interface) + (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul((np.eye(3)-tangeff_dyd_tangeff), r_jump_interface))
                        cohesive_forces = cohesive_axial_forces + interface_constrained_shear_forces
                        cohesive_bending_moments = self.material.compute_cohesive_bending_moments(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max=self.internal_variables[i:i+1, 2:3], position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                        if (update_internal):
                            # update the maximum effective separation
                            new_delta_max = self.material.compute_effective_maximum_separation(delta, delta_max=self.internal_variables[i:i+1, 2:3])
                            self.internal_variables[i:i+1, 2:3] = new_delta_max
            # DG FLUX AND COMPATIBILITY TERMS
            f[global_element_dofs_left] += (1.0-self.internal_variables[i:i+1, 1:2])*(np.matmul(np.transpose(N_left_interface), average_forces_interface) + np.matmul(np.transpose(Np_left_interface), average_mxt4_interface) + (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), r_jump_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_left_interface), rp_jump_interface)))
            f[global_element_dofs_right] -= (1.0-self.internal_variables[i:i+1, 1:2])*(np.matmul(np.transpose(N_right_interface), average_forces_interface) + np.matmul(np.transpose(Np_right_interface), average_mxt4_interface) + (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), r_jump_interface)) + (self.betaT*((self.material.E*self.material.I)/self.function_space.elL)*np.matmul(np.transpose(Np_right_interface), rp_jump_interface)))
            # INTERFACE FORCES AND MOMENTS FROM THE CZM
            f[global_element_dofs_left] += (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(N_left_interface), cohesive_forces)))
            f[global_element_dofs_right] -= (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(N_right_interface), cohesive_forces)))
            f[global_element_dofs_left] += (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(Np_left_interface), cohesive_bending_moments)))
            f[global_element_dofs_right] -= (self.internal_variables[i:i+1, 1:2]*(np.matmul(np.transpose(Np_right_interface), cohesive_bending_moments)))
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
    def compute_system_stiffness(self, A, system_unknowns, element_loads_info):
        # compute system stiffness using the function in WeakFormCG
        super().compute_system_stiffness(A, system_unknowns, element_loads_info)
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
            # initialize the cohesive force derivative coefficients at the interface
            dfcoh_axial_dd_N_dual_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dfcoh_axial_dd_Np_direct_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dfcoh_axial_dd_Np_dual_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dmcoh_bending_dd_N_dual_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dmcoh_bending_dd_Np_direct_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            dmcoh_bending_dd_Np_dual_coeff = np.zeros(dt1dd_Np_left_interface.shape)
            # perform CZM calculations if needed in the case of a cohesive interface material
            if ((isinstance(self.material, (Material.CohesiveInterfaceMaterial))) and (self.internal_variables[i:i+1, 1:2] == 1.0)):
                # compute cohesive axial forces and bending moments derivative coefficients
                dfcoh_axial_dd_N_dual_coeff, dfcoh_axial_dd_Np_direct_coeff, dfcoh_axial_dd_Np_dual_coeff = self.material.compute_cohesive_axial_forces_derivative_coefficients(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max=self.internal_variables[i:i+1, 2:3], position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                dmcoh_bending_dd_N_dual_coeff, dmcoh_bending_dd_Np_direct_coeff, dmcoh_bending_dd_Np_dual_coeff = self.material.compute_cohesive_bending_moments_derivative_coefficients(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max=self.internal_variables[i:i+1, 2:3], position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                # effective separation at the interface
                delta = self.material.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=self.internal_variables[i:i+1, 4:7].T, tangent_jumps_DI=self.internal_variables[i:i+1, 7:10].T)
                # constrained shear forces derivatives
                if ((delta < self.material.delta_c) and (self.internal_variables[i:i+1, 2:3] < self.material.delta_c)): # before fully developed fracture
                    effective_unit_tangent_interface = self.material.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
                    tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
                    # the dual terms
                    A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), N_left_interface)))
                    A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), N_right_interface)))
                    A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_left_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), N_right_interface)))
                    A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= (self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(np.transpose(N_right_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), N_left_interface)))
                    t1_left_interface, _, _, _, t5_left_interface = self.compute_residual_vectors(rp_left_interface, rpp_left_interface, rppp_left_interface)
                    t1_right_interface, _, _, _, t5_right_interface = self.compute_residual_vectors(rp_right_interface, rpp_right_interface, rppp_right_interface)
                    # forces at the interface
                    # works only when there are no elemental loads!!!
                    forces_left_interface = (self.material.E*self.material.A*t1_left_interface) + (self.material.E*self.material.I*t5_left_interface)
                    forces_right_interface = (self.material.E*self.material.A*t1_right_interface) + (self.material.E*self.material.I*t5_right_interface)
                    average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
                    # the first set of direct terms
                    A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= 0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_left_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt1dd_left_interface))) + (self.material.E*self.material.I*np.matmul(np.transpose(N_left_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt5dd_left_interface))))
                    A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += 0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_right_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt1dd_left_interface))) + (self.material.E*self.material.I*np.matmul(np.transpose(N_right_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt5dd_left_interface))))
                    A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= 0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_left_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt1dd_right_interface))) + (self.material.E*self.material.I*np.matmul(np.transpose(N_left_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt5dd_right_interface))))
                    A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += 0.50*((self.material.E*self.material.A*np.matmul(np.transpose(N_right_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt1dd_right_interface))) + (self.material.E*self.material.I*np.matmul(np.transpose(N_right_interface), np.matmul((np.eye(self.function_space.dim) - tangeff_dyd_tangeff), dt5dd_right_interface))))
                    # second set of direct terms
                    dfDG_perp_dd_term2_Np_direct_coeff, dcDG_perp_dd_term2_Np_direct_coeff = self.material.compute_constrained_shear_forces_derivative_coefficients(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, average_forces_interface, position_jumps_DI=self.internal_variables[i:i+1, 4:7].T)
                    dfDG_perp_dd_term2_direct_left = np.matmul(dfDG_perp_dd_term2_Np_direct_coeff, Np_left_interface)
                    dfDG_perp_dd_term2_direct_right = np.matmul(dfDG_perp_dd_term2_Np_direct_coeff, Np_right_interface)
                    dfcomp_perp_dd_term2_direct_left = self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(dcDG_perp_dd_term2_Np_direct_coeff, Np_left_interface)
                    dfcomp_perp_dd_term2_direct_right = self.betaP*((self.material.E*self.material.A)/self.function_space.elL)*np.matmul(dcDG_perp_dd_term2_Np_direct_coeff, Np_right_interface)
                    dfcons_shear_dd_term2_direct_left = dfDG_perp_dd_term2_direct_left + dfcomp_perp_dd_term2_direct_left
                    dfcons_shear_dd_term2_direct_right = dfDG_perp_dd_term2_direct_right + dfcomp_perp_dd_term2_direct_right
                    A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= np.matmul(np.transpose(N_left_interface), dfcons_shear_dd_term2_direct_left)
                    A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += np.matmul(np.transpose(N_right_interface), dfcons_shear_dd_term2_direct_left)
                    A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= np.matmul(np.transpose(N_left_interface), dfcons_shear_dd_term2_direct_right)
                    A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += np.matmul(np.transpose(N_right_interface), dfcons_shear_dd_term2_direct_right)
                # update the maximum effective separation
                new_delta_max = self.material.compute_effective_maximum_separation(delta, delta_max=self.internal_variables[i:i+1, 2:3])
                self.internal_variables[i:i+1, 2:3] = new_delta_max
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
            # cohesive axial force and bending moment derivatives from the CZM
            # the dual cohesive axial force terms
            dfcoh_axial_dd_dual_left = np.matmul(dfcoh_axial_dd_N_dual_coeff, N_left_interface) + np.matmul(dfcoh_axial_dd_Np_dual_coeff, Np_left_interface)
            dfcoh_axial_dd_dual_right = np.matmul(dfcoh_axial_dd_N_dual_coeff, N_right_interface) + np.matmul(dfcoh_axial_dd_Np_dual_coeff, Np_right_interface)
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_left_interface), dfcoh_axial_dd_dual_left))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_right_interface), dfcoh_axial_dd_dual_right))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_left_interface), dfcoh_axial_dd_dual_right))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_right_interface), dfcoh_axial_dd_dual_left))
            # the direct cohesive axial force terms
            dfcoh_axial_dd_direct_left = np.matmul(dfcoh_axial_dd_Np_direct_coeff, Np_left_interface)
            dfcoh_axial_dd_direct_right = np.matmul(dfcoh_axial_dd_Np_direct_coeff, Np_right_interface)
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_left_interface), dfcoh_axial_dd_direct_left))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_right_interface), dfcoh_axial_dd_direct_left))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_left_interface), dfcoh_axial_dd_direct_right))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(N_right_interface), dfcoh_axial_dd_direct_right))
            # the dual cohesive bending moment terms
            dmcoh_bending_dd_dual_left = np.matmul(dmcoh_bending_dd_N_dual_coeff, N_left_interface) + np.matmul(dmcoh_bending_dd_Np_dual_coeff, Np_left_interface)
            dmcoh_bending_dd_dual_right = np.matmul(dmcoh_bending_dd_N_dual_coeff, N_right_interface) + np.matmul(dmcoh_bending_dd_Np_dual_coeff, Np_right_interface)
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_left_interface), dmcoh_bending_dd_dual_left))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_right_interface), dmcoh_bending_dd_dual_right))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_left_interface), dmcoh_bending_dd_dual_right))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_right_interface), dmcoh_bending_dd_dual_left))
            # the direct cohesive bending moment terms
            dmcoh_bending_dd_direct_left = np.matmul(dmcoh_bending_dd_Np_direct_coeff, Np_left_interface)
            dmcoh_bending_dd_direct_right = np.matmul(dmcoh_bending_dd_Np_direct_coeff, Np_right_interface)
            A[np.ix_(global_element_dofs_left, global_element_dofs_left)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_left_interface), dmcoh_bending_dd_direct_left))
            A[np.ix_(global_element_dofs_right, global_element_dofs_left)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_right_interface), dmcoh_bending_dd_direct_left))
            A[np.ix_(global_element_dofs_left, global_element_dofs_right)] -= (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_left_interface), dmcoh_bending_dd_direct_right))
            A[np.ix_(global_element_dofs_right, global_element_dofs_right)] += (self.internal_variables[i:i+1, 1:2]*np.matmul(np.transpose(Np_right_interface), dmcoh_bending_dd_direct_right))
        pass

    # Function to compute the system nodal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads_info=None):
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
            t1_left_node, _, _, _, t5_left_node = self.compute_residual_vectors(rp_left_node, rpp_left_node, rppp_left_node)
            t1_right_node, _, _, _, t5_right_node = self.compute_residual_vectors(rp_right_node, rpp_right_node, rppp_right_node)
            # forces at the nodes
            if (element_loads_info == None): # No element loads
                forces_left_node = (self.material.E*self.material.A*t1_left_node) + (self.material.E*self.material.I*t5_left_node)
                forces_right_node = (self.material.E*self.material.A*t1_right_node) + (self.material.E*self.material.I*t5_right_node)
            # moments at the nodes
            moments_left_node = self.material.E*self.material.I*(cross_op(rp_left_node, rpp_left_node, 0, 0, 0)/(rp_left_node_L2**2.0))
            moments_right_node = self.material.E*self.material.I*(cross_op(rp_right_node, rpp_right_node, 0, 0, 0)/(rp_right_node_L2**2.0))
            f[global_element_dofs_left_node[0:int(dofs/2)]] += forces_left_node
            f[global_element_dofs_left_node[int(dofs/2):dofs]] += moments_left_node
            f[global_element_dofs_right_node[0:int(dofs/2)]] += forces_right_node
            f[global_element_dofs_right_node[int(dofs/2):dofs]] += moments_right_node
        pass

class EulerBernoulliWeakFormCG(WeakFormCG):
    
    def __init__(self, function_space, material):
        # invoke the parent (WeakFormCG) class
        WeakFormCG.__init__(self, function_space, material)

    # Function to compute element internal forces
    def compute_element_internal_forces(self, element_unknowns, boundary_dof_jumps, \
                                                                extended_boundary_dof_jumps):
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        axial_lifting_shapes = self.function_space.axial_lifting_shape_functions * \
                                                (1.0/self.function_space.jacobian)
        bending_lifting_shapes = self.function_space.bending_lifting_shape_functions * \
                                                ((1.0/self.function_space.jacobian)**2.0)
        # we need the jacobian vector to convert the rotation into derivative of transverse 
        # displacement w.r.t the parametric coordinate in the lifting computation!
        # this operation will change if the jacobian on the either sides of the interface is 
        # different!!!
        jacobian_vector = np.ones([self.function_space.dof*self.function_space.npel, 1])
        jacobian_vector[2:3, 0:1] = self.function_space.jacobian
        jacobian_vector[5:6, 0:1] = self.function_space.jacobian
        ux = np.matmul(phix, element_unknowns)
        wxx = np.matmul(Nxx, element_unknowns)
        # the lifting related part of the axial and bending dof derivatives
        ux += np.matmul(axial_lifting_shapes, boundary_dof_jumps)
        wxx += np.matmul(bending_lifting_shapes, boundary_dof_jumps*jacobian_vector)
        wxx += np.matmul(bending_lifting_shapes, extended_boundary_dof_jumps)
        integrand = self.material.E*self.material.A*np.matmul(np.transpose(phix, axes=(0, 2, 1)), ux) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(Nxx, axes=(0, 2, 1)), wxx)
        return np.sum(integrand*self.function_space.JxW, axis=0, keepdims=False)
    
    # Function to compute element lifting forces
    def compute_element_lifting_forces(self, element_unknowns, boundary_dof_jumps, \
                                                               extended_boundary_dof_jumps):
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        axial_lifting_shapes = self.function_space.axial_lifting_shape_functions * \
                                                (1.0/self.function_space.jacobian)
        bending_lifting_shapes = self.function_space.bending_lifting_shape_functions * \
                                                ((1.0/self.function_space.jacobian)**2.0)
        # jacobian vector to convert the rotation into derivative of transverse displacement 
        # w.r.t the parametric coordinate in the lifting computation!
        jacobian_vector = np.ones([self.function_space.dof*self.function_space.npel, 1])
        jacobian_vector[2:3, 0:1] = self.function_space.jacobian
        jacobian_vector[5:6, 0:1] = self.function_space.jacobian
        ux = np.matmul(phix, element_unknowns) + np.matmul(axial_lifting_shapes, boundary_dof_jumps)
        wxx = np.matmul(Nxx, element_unknowns) + np.matmul(bending_lifting_shapes, \
                                                           boundary_dof_jumps*jacobian_vector)
        wxx += np.matmul(bending_lifting_shapes, extended_boundary_dof_jumps)
        # the axial and bending variational lifting term
        lifting_integrand = self.material.E*self.material.A*np.matmul(np.transpose(axial_lifting_shapes, \
                                                        axes=(0, 2, 1)), ux) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(bending_lifting_shapes * \
                                                        (jacobian_vector.T), axes=(0, 2, 1)), wxx)
        return np.sum(lifting_integrand*self.function_space.JxW, axis=0, keepdims=False)

    # Function to compute the overall system residual
    def compute_system_residual(self, f, system_unknowns, element_loads_info):
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            boundary_dof_jumps = np.zeros([dofspel, 1])
            extended_boundary_dof_jumps = np.zeros([dofspel, 1])
            # the extended boundary dof jumps container is filled in such a way that they can be 
            # interpolated with bending lifting shape functions
            # the expressions used for extended boundary dof jumps and the variational coefficients 
            # are evaluated for quadratic lifting shape functions!!! these expressions have to be 
            # changed if the order of lifting shape functions are changed
            if (i == 0): # left most element
                right_element_dofs = self.function_space.global_connectivity[i+1:i+2].flatten()
                right_next_element_dofs = self.function_space.global_connectivity[i+2:i+3].flatten()
                # jumps at the right boundary
                boundary_dof_jumps[dofs:dofspel, 0:1] = system_unknowns[right_element_dofs[0:dofs]] - \
                                                    element_unknowns[dofs:dofspel]
                extended_boundary_dof_jumps[5:6, 0:1] = 0.75*(system_unknowns[right_next_element_dofs[1:2]] - \
                                                        system_unknowns[right_element_dofs[4:5]])
            elif (i == self.function_space.E-1): # right most element
                left_element_dofs = self.function_space.global_connectivity[i-1:i].flatten()
                left_previous_element_dofs = self.function_space.global_connectivity[i-2:i-1].flatten()
                # jumps at the left boundary
                boundary_dof_jumps[0:dofs, 0:1] = element_unknowns[0:dofs] - \
                                                system_unknowns[left_element_dofs[dofs:dofspel]]
                extended_boundary_dof_jumps[2:3, 0:1] = -0.75*(system_unknowns[left_element_dofs[1:2]] - \
                                                        system_unknowns[left_previous_element_dofs[4:5]])
            else: # intermediate elements
                left_element_dofs = self.function_space.global_connectivity[i-1:i].flatten()
                right_element_dofs = self.function_space.global_connectivity[i+1:i+2].flatten()
                # dof jumps at the boundary
                boundary_dof_jumps[0:dofs, 0:1] = element_unknowns[0:dofs] - \
                                                system_unknowns[left_element_dofs[dofs:dofspel]]
                # jumps at the right boundary
                boundary_dof_jumps[dofs:dofspel, 0:1] = system_unknowns[right_element_dofs[0:dofs]] - \
                                                    element_unknowns[dofs:dofspel]
                if (i == 1): # left last but one element
                    right_next_element_dofs = self.function_space.global_connectivity[i+2:i+3].flatten()
                    extended_boundary_dof_jumps[2:3, 0:1] = 0.75*(system_unknowns[right_element_dofs[1:2]] - \
                                                                  element_unknowns[4:5])
                    extended_boundary_dof_jumps[5:6, 0:1] = 0.75*(system_unknowns[right_next_element_dofs[1:2]] - \
                                                            system_unknowns[right_element_dofs[4:5]]) - \
                                                            0.75*(element_unknowns[1:2] - \
                                                            system_unknowns[left_element_dofs[4:5]])
                elif (i == self.function_space.E-2): # right last but one element
                    left_previous_element_dofs = self.function_space.global_connectivity[i-2:i-1].flatten()
                    extended_boundary_dof_jumps[2:3, 0:1] = 0.75*(system_unknowns[right_element_dofs[1:2]] - \
                                                            element_unknowns[4:5]) - \
                                                            0.75*(system_unknowns[left_element_dofs[1:2]] - \
                                                            system_unknowns[left_previous_element_dofs[4:5]])
                    extended_boundary_dof_jumps[5:6, 0:1] = -0.75*(element_unknowns[1:2] - \
                                                            system_unknowns[left_element_dofs[4:5]])
                else: # other elements
                    left_previous_element_dofs = self.function_space.global_connectivity[i-2:i-1].flatten()
                    right_next_element_dofs = self.function_space.global_connectivity[i+2:i+3].flatten()
                    extended_boundary_dof_jumps[2:3, 0:1] = 0.75*(system_unknowns[right_element_dofs[1:2]] - \
                                                            element_unknowns[4:5]) - \
                                                            0.75*(system_unknowns[left_element_dofs[1:2]] - \
                                                            system_unknowns[left_previous_element_dofs[4:5]])
                    extended_boundary_dof_jumps[5:6, 0:1] = 0.75*(system_unknowns[right_next_element_dofs[1:2]] - \
                                                            system_unknowns[right_element_dofs[4:5]]) - \
                                                            0.75*(element_unknowns[1:2] - \
                                                            system_unknowns[left_element_dofs[4:5]])
            if (element_loads_info == None): # No element loads
                element_internal_forces = \
                    self.compute_element_internal_forces(element_unknowns, boundary_dof_jumps, \
                                                         extended_boundary_dof_jumps)
                # element internal forces
                f[global_element_dofs] -= element_internal_forces
                # element lifting and extended lifting forces
                if (self.function_space.discretization_type == "DG"):
                    element_lifting_forces = \
                    self.compute_element_lifting_forces(element_unknowns, boundary_dof_jumps, \
                                                        extended_boundary_dof_jumps)
                    if (i == 0): # left most element
                        # right side
                        f[global_element_dofs[dofs:dofspel]] += element_lifting_forces[dofs:dofspel]
                        f[right_element_dofs[0:dofs]] -= element_lifting_forces[dofs:dofspel]
                    elif (i == self.function_space.E-1): # right most element
                        # left side
                        f[global_element_dofs[0:dofs]] -= element_lifting_forces[0:dofs]
                        f[left_element_dofs[dofs:dofspel]] += element_lifting_forces[0:dofs]
                    else: # intermediate elements
                        # left side
                        f[global_element_dofs[0:dofs]] -= element_lifting_forces[0:dofs]
                        f[left_element_dofs[dofs:dofspel]] += element_lifting_forces[0:dofs]
                        # right side
                        f[global_element_dofs[dofs:dofspel]] += element_lifting_forces[dofs:dofspel]
                        f[right_element_dofs[0:dofs]] -= element_lifting_forces[dofs:dofspel]
        pass
    
    # Function to compute element internal stiffness
    def compute_element_internal_stiffness(self, element_unknowns):
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        integrand = self.material.E*self.material.A*np.matmul(np.transpose(phix, axes=(0, 2, 1)), phix) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(Nxx, axes=(0, 2, 1)), Nxx)
        return np.sum(integrand*self.function_space.JxW, axis=0, keepdims=False)
    
    # Function to compute the system mass
    def compute_system_mass(self, M, system_unknowns, use_rotational_mass=False, lump=True):
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
        pass
    
    # Function to compute the system nodal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads_info=None):
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
        pass

class EulerBernoulliWeakFormDG(EulerBernoulliWeakFormCG):
    
    def __init__(self, function_space, material, beta):
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
    def compute_interface_forces(self, element_unknowns_left, element_unknowns_right):
        # dofs and their derivaitives at the interface
        # left side
        u_left = np.matmul(self.phi_left_interface, element_unknowns_left)
        w_left = np.matmul(self.N_left_interface, element_unknowns_left)
        wx_left = np.matmul(self.Nx_left_interface, element_unknowns_left)
        shear_force_left = -self.material.E*self.material.I * \
                                np.matmul(self.Nxxx_left_interface, element_unknowns_left)
        bending_moment_left = -self.material.E*self.material.I * \
                                np.matmul(self.Nxx_left_interface, element_unknowns_left)
        # right side
        u_right = np.matmul(self.phi_right_interface, element_unknowns_right)
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
        axial_forces_interface = (self.beta*((self.material.E*self.material.A) / \
                                                   self.function_space.elL)*u_jump)
        shear_forces_interface = ((shear_force_left + shear_force_right) / 2.0) + \
                                        (self.beta*((self.material.E*self.material.A) / \
                                                   self.function_space.elL)*w_jump)
        bending_moments_interface = -((bending_moment_left + bending_moment_right) / 2.0) + \
                                        (self.beta*((self.material.E*self.material.I) / \
                                                   self.function_space.elL)*wx_jump)
        return axial_forces_interface, shear_forces_interface, bending_moments_interface

    # Function to compute the system residual
    def compute_system_residual(self, f, system_unknowns, element_loads_info):
        # compute system residual using the function in EulerBernoulliWeakFormCG
        super().compute_system_residual(f, system_unknowns, element_loads_info)
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
                self.compute_interface_forces(element_unknowns_left, element_unknowns_right)
            # assemble the interface forces
            f[global_element_dofs_left] += \
                    np.matmul(np.transpose(self.phi_left_interface), axial_forces_interface) + \
                    np.matmul(np.transpose(self.N_left_interface), shear_forces_interface) + \
                    np.matmul(np.transpose(self.Nx_left_interface), bending_moments_interface)
            f[global_element_dofs_right] -= \
                    np.matmul(np.transpose(self.phi_right_interface), axial_forces_interface) + \
                    np.matmul(np.transpose(self.N_right_interface), shear_forces_interface) + \
                    np.matmul(np.transpose(self.Nx_right_interface), bending_moments_interface)
        pass

    # Function to compute the system stiffness
    def compute_system_stiffness(self, A, system_unknowns, element_loads_info):
        # compute system stiffness using the function in EulerBernoulliWeakFormCG
        super().compute_system_stiffness(A, system_unknowns, element_loads_info)
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
        pass

    # Function to compute the system nodal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads_info=None):
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
        pass