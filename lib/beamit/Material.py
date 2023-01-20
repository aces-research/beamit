import numpy as np

class Material:
    
    def __init__(self, rho, E, A, I):
        # the density
        self.rho = rho
        # the elastic modulus
        self.E = E
        # the area of the beam cross section
        self.A = A
        # the area moment of inertia
        self.I = I
        print("\nCreated the material.")

class CohesiveInterfaceMaterial(Material):

    def __init__(self, rho, E, A, I, Sc, Gc, gamma = 1.0):
        # invoke the parent (Material) class
        Material.__init__(self, rho, E, A, I)
        # critical effective cohesive strength of the material
        self.Sc = Sc
        # effective fracture energy of the material
        self.Gc = Gc
        # weighting parameter
        self.gamma = gamma
        # critical effective separation
        self.delta_c = (2.0*self.Gc)/self.Sc
        # critical effective force
        self.fc = self.Sc*self.A
        # constant for non-dimensionalization
        self.C = np.sqrt(self.A/np.pi)
        # the mode-mixity parameter
        self.alpha = 1.0

    # Function to compute the effective unit tangent ("normal to the cohesive boundary") at an interface
    def compute_effective_unit_tangent(self, rp_left_interface, rp_right_interface):
        tangent_left_interface = rp_left_interface/np.linalg.norm(rp_left_interface, ord=2, axis=0, keepdims=True)
        tangent_right_interface = rp_right_interface/np.linalg.norm(rp_right_interface, ord=2, axis=0, keepdims=True)
        average_tangent_interface = (tangent_left_interface + tangent_right_interface)/2.0
        effective_unit_tangent_interface = average_tangent_interface/np.linalg.norm(average_tangent_interface, ord=2, axis=0, keepdims=True)
        return effective_unit_tangent_interface
    
    # Function to compute effective force at an interface
    def compute_effective_force(self, rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface):
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
        average_axial_force_interface = np.sum(average_forces_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial forces are tensile
        average_tensile_force_interface = np.maximum(average_axial_force_interface, 0.0)
        # tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
        average_bending_moments_interface = (moments_left_interface + moments_right_interface)/2.0
        # average_bending_moments_interface = np.matmul((np.eye(3)-tangeff_dyd_tangeff), average_moments_interface)
        average_bending_moments_interface_L2 = np.linalg.norm(average_bending_moments_interface, ord=2, axis=0, keepdims=True)
        # considering mixed-mode fracture with tensile forces and bending moments
        effective_force_interface = np.sqrt((average_tensile_force_interface**2.0)+((average_bending_moments_interface_L2/(self.alpha*self.C))**2.0))
        return effective_force_interface
    
    # Function to evaluate the damage initiation criterion at an interface
    def evaluate_damage_initiation_criterion(self, rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface):
        effective_force_interface = self.compute_effective_force(rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface)
        # if the effective force at the interface satisfies the damage initiation criterion
        if (effective_force_interface/self.fc >= 1.0):
            return True
        else:
            return False

    # Function to compute the effective separation across the "cohesive boundary"
    def compute_effective_separation(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=np.zeros([3,1])):
        # compute the effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface - position_jumps_DI
        axial_jump = np.sum(r_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial jump is tensile
        tensile_jump = np.maximum(axial_jump, 0.0)
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface
        tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
        # bending rotations at the interface
        bending_rotations = np.matmul((np.eye(3)-tangeff_dyd_tangeff), rp_jump_interface)
        bending_rotations_L2 = np.linalg.norm(bending_rotations, ord=2, axis=0, keepdims=True)
        # considering mixed-mode fracture with tensile jumps and bending rotations
        delta = np.sqrt((tensile_jump**2.0)+((self.alpha*self.C*bending_rotations_L2)**2.0))
        return delta

    # Function to compute the axial separation
    def compute_axial_separation(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=np.zeros([3,1])):
        # compute the effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface - position_jumps_DI
        axial_jump = np.sum(r_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        return axial_jump
    
    # Function to compute the "new" maximum effective separation
    def compute_effective_maximum_separation(self, delta, delta_max):
        if ((delta >= self.delta_c) or (delta >= delta_max)): # loading or complete damage
            new_delta_max = delta
        else: # unloading
            new_delta_max = delta_max
        return new_delta_max
    
    # Function to compute the effective cohesive force according to a linear TSL
    def compute_effective_cohesive_force(self, delta, delta_max, effective_force_DI=None):
        # set the effective force to critical force in case of no input
        if (effective_force_DI == None):
            effective_force_DI = self.fc
        if (delta >= self.delta_c): # complete damage
            fcoh = 0.0
        elif (delta >= delta_max): # loading
            fcoh = effective_force_DI*(1.0-(delta/self.delta_c))
        else: # unloading
            fmax = effective_force_DI*(1.0-(delta_max/self.delta_c))
            fcoh = (fmax/delta_max)*delta
        return fcoh
    
    # Function to compute the cohesive forces at the interface
    def compute_cohesive_forces(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3,1])):
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface - position_jumps_DI
        axial_jump = np.sum(r_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial jump is tensile
        tensile_jump = np.maximum(axial_jump, 0.0)
        # compute the cohesive forces
        cohesive_forces = (fcoh/delta)*tensile_jump*effective_unit_tangent_interface
        return cohesive_forces

    # Function to compute the cohesive moments at the interface
    def compute_cohesive_moments(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3,1])):
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface
        tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
        # bending rotations at the interface
        bending_rotations = np.matmul((np.eye(3)-tangeff_dyd_tangeff), rp_jump_interface)
        # compute the cohesive moments
        cohesive_moments = (fcoh/delta)*((self.alpha*self.C)**2.0)*bending_rotations
        return cohesive_moments
    
    # Function to compute the derivative of the effective cohesive force w.r.t the effective separation
    def compute_effective_cohesive_force_derivative(self, delta, delta_max, effective_force_DI=None):
        # set the effective force to critical effective force in case of no input
        if (effective_force_DI == None):
            effective_force_DI = self.fc
        if (delta >= self.delta_c): # complete damage
            dfcoh_ddelta = 0.0
        elif (delta >= delta_max): # loading
            dfcoh_ddelta = -effective_force_DI/self.delta_c
        else: # unloading
            fmax = effective_force_DI*(1.0-(delta_max/self.delta_c))
            dfcoh_ddelta = fmax/delta_max
        return dfcoh_ddelta
    
    # Function to compute the coefficients of the cohesive force derivatives
    def compute_cohesive_force_derivative_coefficients(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3,1])):
        tangent_left_interface = rp_left_interface/np.linalg.norm(rp_left_interface, ord=2, axis=0, keepdims=True)
        tangent_right_interface = rp_right_interface/np.linalg.norm(rp_right_interface, ord=2, axis=0, keepdims=True)
        average_tangent_interface = (tangent_left_interface + tangent_right_interface)/2.0
        average_tangent_interface_L2 = np.linalg.norm(average_tangent_interface, ord=2, axis=0, keepdims=True)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface - position_jumps_DI
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI)
        # effective cohesive force and its derivative w.r.t effective separation at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        dfcoh_ddelta = self.compute_effective_cohesive_force_derivative(delta, delta_max, effective_force_DI)
        avtan_dyd_avtan = np.matmul(average_tangent_interface, np.transpose(average_tangent_interface))     
        dtangeffdd_coeff = 0.50*((np.eye(3)/average_tangent_interface_L2) - (avtan_dyd_avtan/(average_tangent_interface_L2**3.0)))
        tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
        tangeff_dyd_rjump = np.matmul(effective_unit_tangent_interface, np.transpose(r_jump_interface))
        # first term coefficients
        dft1dd_N_coeff = dfcoh_ddelta*tangeff_dyd_tangeff
        dft1dd_Np_coeff = dfcoh_ddelta*np.matmul(tangeff_dyd_rjump, dtangeffdd_coeff)
        # second term coefficient
        dft2dd_Np_coeff = fcoh*dtangeffdd_coeff
        return dft1dd_N_coeff, dft1dd_Np_coeff, dft2dd_Np_coeff
        