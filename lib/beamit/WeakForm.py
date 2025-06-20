from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import quaternion
from beamit import Material

def skew_symmetric_matrices(vectors):
        """
        Given an (n, 3) array of **vectors**, return an (n, 3, 3) array of their skew-symmetric matrix form.
        """
        # check if vectors have shape (n, 3)
        if (vectors.ndim != 2 or vectors.shape[1] != 3):
            raise ValueError("Input must be an (n, 3) array of vectors.")

        n = vectors.shape[0]
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]

        skew = np.zeros((n, 3, 3), dtype=vectors.dtype)
        skew[:, 0, 1] = -z
        skew[:, 0, 2] = y
        skew[:, 1, 0] = z
        skew[:, 1, 2] = -x
        skew[:, 2, 0] = -y
        skew[:, 2, 1] = x

        return skew

class SolutionUpdateType(Enum):
    """
    Enum for the type of update to be applied to the solution.

    Attributes:
        ADD_TNS_ADD_ROT: Additive update to both translations and rotations.
        ADD_TNS_MUL_ROT: Additive update to translations and multiplicative update to rotations.
    """
    ADD_TNS_ADD_ROT = 1
    ADD_TNS_MUL_ROT = 2

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

    def update_bulk_internal_variables(self, system_unknowns_increment):
        """
        Update the internal variables in the weak form based on the increment in the system unknowns.

        Parameters:
            system_unknowns_increment: The increment in the system unknowns.
        """
        pass

class TFKLGeometricallyExactWeakFormCG(WeakForm):
    
    def __init__(self, function_space, material):
        # initialize the parent (WeakForm) class
        WeakForm.__init__(self, function_space, material)

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

    # Function to add nodal loads to the residual
    def add_nodal_loads_to_residual(self, f, system_unknowns, nodal_loads):
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
            t1_right_node, _, _, _, t5_right_node = self._compute_residual_vectors(
                rp_right_node, rpp_right_node, rppp_right_node)
            # forces at the right node
            if (element_loads == None): # No element loads
                forces_right_node = (self.material.E*self.material.A*t1_right_node) + (self.material.E*self.material.I*t5_right_node)
            # moments at the right node
            moments_right_node = self.material.E*self.material.I*(np.cross(rp_right_node, rpp_right_node, axis=0)/(rp_right_node_L2**2.0))
            f[global_element_dofs_right_node[0:int(dofs/2)]] += forces_right_node
            f[global_element_dofs_right_node[int(dofs/2):dofs]] += moments_right_node
            # Assuming the elements are connected like a simple chain!!!
            if (i == 0): # only for the first element
                global_element_dofs_left_node = global_element_dofs[0:dofs]
                rp_left_node = np.matmul(Np_left_node, element_unknowns)
                rpp_left_node = np.matmul(Npp_left_node, element_unknowns)
                rppp_left_node = np.matmul(Nppp_left_node, element_unknowns)
                rp_left_node_L2 = np.linalg.norm(rp_left_node, ord=2, axis=0, keepdims=True)
                t1_left_node, _, _, _, t5_left_node = self._compute_residual_vectors(
                    rp_left_node, rpp_left_node, rppp_left_node)
                # forces at the left node
                if (element_loads == None): # No element loads
                    forces_left_node = (self.material.E*self.material.A*t1_left_node) + (self.material.E*self.material.I*t5_left_node)
                # moments at the left node
                moments_left_node = self.material.E*self.material.I*(np.cross(rp_left_node, rpp_left_node, axis=0)/(rp_left_node_L2**2.0))
                f[global_element_dofs_left_node[0:int(dofs/2)]] += forces_left_node
                f[global_element_dofs_left_node[int(dofs/2):dofs]] += moments_left_node

    # Function to compute the system mass
    def compute_system_mass(self, M, system_unknowns, lump=True):
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
            M[np.ix_(global_element_dofs, global_element_dofs)] += M_el_trans

class TFKLGeometricallyExactWeakFormDG(TFKLGeometricallyExactWeakFormCG):

    def __init__(self, function_space, material, betaP, betaT):
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

    # Function to compute the system nodal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads):
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

class EulerBernoulliWeakFormCG(WeakForm):
    
    def __init__(self, function_space, material):
        # initialize the parent (WeakForm) class
        WeakForm.__init__(self, function_space, material)

    # Function to compute element internal forces
    def compute_element_internal_forces(self, element_unknowns):
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        ux = np.matmul(phix, element_unknowns)
        wxx = np.matmul(Nxx, element_unknowns)
        integrand = self.material.E*self.material.A*np.matmul(np.transpose(phix, axes=(0, 2, 1)), ux) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(Nxx, axes=(0, 2, 1)), wxx)
        return np.sum(integrand*self.function_space.JxW, axis=0, keepdims=False)

    # Function to compute element internal stiffness
    def compute_element_internal_stiffness(self, element_unknowns):
        phix = self.function_space.lagrange_shape_first_gradients*(1.0/self.function_space.jacobian)
        Nxx = self.function_space.hermite_shape_second_gradients*((1.0/self.function_space.jacobian)**2.0)
        integrand = self.material.E*self.material.A*np.matmul(np.transpose(phix, axes=(0, 2, 1)), phix) + \
                            self.material.E*self.material.I*np.matmul(np.transpose(Nxx, axes=(0, 2, 1)), Nxx)
        return np.sum(integrand*self.function_space.JxW, axis=0, keepdims=False)

    # Function to compute the system mass
    def compute_system_mass(self, M, system_unknowns, lump=True):
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

class ShearFlexibleGeometricallyExactWeakFormCG(WeakForm):

    def __init__(self, function_space, material):
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
        psi_norm_safe = np.where(psi_norm < 1.0e-10, 1.0e-10, psi_norm)
        sin_psi = np.sin(psi_norm)
        cos_psi = np.cos(psi_norm)
        sin_term = (sin_psi / psi_norm_safe)[:, None, None]
        cos_term = ((1.0 - cos_psi) / psi_norm_safe**2)[:, None, None]
        extra_term = ((psi_norm - sin_psi) / psi_norm_safe**3)[:, None, None]
        # skew symmetric matrices
        S = skew_symmetric_matrices(orientation)
        # outer product
        outer = orientation[:, :, None] * orientation[:, None, :]
        return sin_term * I + cos_term * S + extra_term * outer

    def update_bulk_internal_variables(self, system_unknowns_increment):
        """
        Update the internal variables in the weak form based on the increment in the system unknowns.
        
        The orientation and curvature of the beam at each quadrature point are updated based on 
        the current system unknowns increment.

        Parameters:
            system_unknowns_increment: The increment in the system unknowns.
        """
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_rotation_increment = system_unknowns_increment[
                global_element_dofs][self.function_space.local_rotational_dofs]
            # compute the orientation increment and its derivative
            dtheta = np.matmul(N, element_rotation_increment)[..., 0]
            dtheta_prime = np.matmul(Np, element_rotation_increment)[..., 0]
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

    def __compute_internal_forces_and_moments(self, element_unknowns, element_orientations, 
                                              element_curvatures):
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        ################# internal forces #################
        rp = np.matmul(
            Np, element_unknowns[self.function_space.local_translational_dofs])
        # check if the element orientations are of size (Q, dim)
        if (element_orientations.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element orientations must be of shape (Q, dim)")
        element_orientations_tensor = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(element_orientations))
        E1 = np.zeros([self.function_space.Q, self.function_space.dim])
        E1[:, 0] = 1.0
        element_strains = rp - \
            np.matmul(element_orientations_tensor, E1[..., None])
        # translation constitutive matrix
        C_F = np.zeros(
            [self.function_space.Q, self.function_space.dim, self.function_space.dim])
        C_F[:, 0, 0] = self.material.E * self.material.A
        C_F[:, 1, 1] = self.material.G * self.material.A_red
        C_F[:, 2, 2] = self.material.G * self.material.A_red
        # internal forces
        C_F_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_F, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        internal_forces = np.matmul(C_F_transformed, element_strains)
        ################# internal moments #################
        # check if the element curvatures are of size (Q, dim)
        if (element_curvatures.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element curvatures must be of shape (Q, dim)")
        # rotational constitutive matrix
        C_M = np.zeros(
            [self.function_space.Q, self.function_space.dim, self.function_space.dim])
        C_M[:, 0, 0] = self.material.G * self.material.I_T
        C_M[:, 1, 1] = self.material.E * self.material.I
        C_M[:, 2, 2] = self.material.E * self.material.I_minor
        # NOTE: Since the beam is initially straight, there is no initial curvature!!!
        C_M_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_M, np.transpose(element_orientations_tensor, axes=(0, 2, 1))))
        internal_moments = np.matmul(
            C_M_transformed, element_curvatures[..., None])
        return internal_forces, internal_moments

    def compute_element_internal_forces(self, element_unknowns, element_orientations, 
                                        element_curvatures):
        element_internal_forces = np.zeros(
            [self.function_space.dof*self.function_space.npel, 1])
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        rp = np.matmul(
            Np, element_unknowns[self.function_space.local_translational_dofs])
        # compute the internal forces and moments
        internal_forces, internal_moments = self.__compute_internal_forces_and_moments(
            element_unknowns, element_orientations, element_curvatures)
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

    def __compute_element_material_stiffness(self, element_unknowns, element_orientations):
        element_material_stiffness = np.zeros(
            [self.function_space.dof*self.function_space.npel, 
             self.function_space.dof*self.function_space.npel])
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        rp = np.matmul(
            Np, element_unknowns[self.function_space.local_translational_dofs])
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

    def __compute_element_geometric_stiffness(self, element_unknowns, element_orientations,
                                              element_curvatures):
        element_geometric_stiffness = np.zeros(
            [self.function_space.dof*self.function_space.npel,
             self.function_space.dof*self.function_space.npel])
        N = self.function_space.shape_functions
        Np = self.function_space.shape_first_gradients * \
            (1.0/self.function_space.jacobian)
        rp = np.matmul(
            Np, element_unknowns[self.function_space.local_translational_dofs])
        # check if the element orientations and curvatures are of size (Q, dim)
        if (element_orientations.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element orientations must be of shape (Q, dim)")
        if (element_curvatures.shape != (self.function_space.Q, self.function_space.dim)):
            raise ValueError("Element curvatures must be of shape (Q, dim)")
        # compute the internal forces and moments
        internal_forces, internal_moments = self.__compute_internal_forces_and_moments(
            element_unknowns, element_orientations, element_curvatures)
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

    def compute_element_internal_stiffness(self, element_unknowns, element_orientations, 
                                           element_curvatures):
        element_internal_stiffness = np.zeros(
            [self.function_space.dof*self.function_space.npel, 
             self.function_space.dof*self.function_space.npel])
        # compute the element material stiffness
        element_internal_stiffness += self.__compute_element_material_stiffness(
            element_unknowns, element_orientations)
        # compute the element geometric stiffness
        element_internal_stiffness += self.__compute_element_geometric_stiffness(
            element_unknowns, element_orientations, element_curvatures)
        return element_internal_stiffness

    def compute_system_residual(self, f, system_unknowns, element_loads, update_internal):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten(
            )
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads == None):  # No element loads
                f[global_element_dofs] -= self.compute_element_internal_forces(
                    element_unknowns, self.orientation[i, :, :], self.curvature[i, :, :])

    def compute_system_stiffness(self, A, system_unknowns, nodal_loads, element_loads):
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten(
            )
            element_unknowns = system_unknowns[global_element_dofs]
            if (element_loads == None):  # No element loads
                A[np.ix_(global_element_dofs, global_element_dofs)
                  ] += self.compute_element_internal_stiffness(element_unknowns, 
                                                               self.orientation[i, :, :], 
                                                               self.curvature[i, :, :])

    # Function to compute the system nodal forces
    # Computed by approaching every node from the left side!!!
    def compute_system_nodal_forces(self, f, system_unknowns, element_loads):
        dofs = self.function_space.dof
        dofspel = self.function_space.dof*self.function_space.npel
        _, Nxi_left_node = self.function_space.compute_shapes(-1.0)
        Np_left_node = Nxi_left_node*(1.0/self.function_space.jacobian)
        _, Nxi_right_node = self.function_space.compute_shapes(1.0)
        Np_right_node = Nxi_right_node*(1.0/self.function_space.jacobian)
        for i in range(0, self.function_space.E):
            global_element_dofs = self.function_space.global_connectivity[i:i+1].flatten()
            element_unknowns = system_unknowns[global_element_dofs]
            global_element_dofs_left_node = global_element_dofs[0:dofs]
            global_element_dofs_right_node = global_element_dofs[dofs:dofspel]
            # IMPLEMENT THIS FUNCTION FOR GETTING FORCE AND MOMENT OUTPUTS LATER!!!
