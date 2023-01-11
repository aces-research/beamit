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

    # Function to compute the effective unit tangent ("normal to the cohesive boundary") at an interface - CHECK!!!
    def compute_effective_unit_tangent(self, rp_left_interface, rp_right_interface):
        average_rp_interface = (rp_left_interface + rp_right_interface)/2.0
        effective_unit_tangent_interface = average_rp_interface/np.linalg.norm(average_rp_interface, ord=2, axis=0, keepdims=True)
        return effective_unit_tangent_interface
    
    # Function to compute effective force at an interface
    def compute_effective_force(self, rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface):
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
        average_axial_force_interface = np.sum(average_forces_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial forces are tensile in nature
        average_tensile_force_interface = np.maximum(average_axial_force_interface, 0.0)
        # considering only axial forces for now!!!
        effective_force_interface = average_tensile_force_interface
        return effective_force_interface
    
    # Function to evaluate the damage initiation criterion at an interface
    def evaluate_damage_initiation_criterion(self, rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface):
        effective_force_interface = self.compute_effective_force(rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface)
        # if the effective force at the interface satisfies the damage initiation criterion
        if (effective_force_interface/self.fc >= 1.0):
            return True
        else:
            return False

    # Function to compute the effective separation across the "cohesive boundary"
    def compute_effective_separation(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface):
        # compute the effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface
        axial_jump = np.sum(r_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial jump is tensile
        tensile_jump = np.maximum(axial_jump, 0.0)
        # considering only tensile jumps for now!!!
        delta = tensile_jump
        return delta

    # Function to compute the axial separation
    def compute_axial_separation(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface):
        # compute the effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface
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
    def compute_cohesive_forces(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None):
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        # effective unit tangent at the interface
        effective_unit_tangent = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # compute the cohesive forces
        cohesive_forces = fcoh*effective_unit_tangent
        return cohesive_forces

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
    def compute_cohesive_force_derivative_coefficients(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None):
        average_rp_interface = (rp_left_interface + rp_right_interface)/2.0
        average_rp_interface_L2 = np.linalg.norm(average_rp_interface, ord=2, axis=0, keepdims=True)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        avrp_dyd_avrp = np.matmul(average_rp_interface, np.transpose(average_rp_interface))
        dtangeffdd_coeff = 0.50*((np.eye(rp_left_interface.shape[0])/average_rp_interface_L2) - (avrp_dyd_avrp/(average_rp_interface_L2**3.0)))
        tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
        tangeff_dyd_rjump = np.matmul(effective_unit_tangent_interface, np.transpose(r_jump_interface))
        dfcoh_ddelta = self.compute_effective_cohesive_force_derivative(delta, delta_max, effective_force_DI)
        # first term coefficients
        dft1dd_N_coeff = dfcoh_ddelta*tangeff_dyd_tangeff
        dft1dd_Np_coeff = dfcoh_ddelta*np.matmul(tangeff_dyd_rjump, dtangeffdd_coeff)
        # second term coefficient
        dft2dd_Np_coeff = fcoh*dtangeffdd_coeff
        return dft1dd_N_coeff, dft1dd_Np_coeff, dft2dd_Np_coeff
        