import numpy as np
import sys

class Material:
    
    def __init__(self, rho, E, R = None, A = None, I = None):
        # the density
        self.rho = rho
        # the elastic modulus
        self.E = E
        if (R != None):
            # the radius of the beam
            self.R = R
            # the area of the beam cross section
            self.A = np.pi*(R**2.0)
            # the area moment of inertia
            self.I = (np.pi*(R**4.0))/4.0
        elif ((A != None) and (I != None)):
            self.R = np.sqrt(A/np.pi)
            self.A = A
            self.I = I
        else:
            sys.exit("\nEither R or A & I has to be given as an input.")
        print("\nCreated the material.")

class CohesiveInterfaceMaterial(Material):

    def __init__(self, rho, E, R, Sc, Gc, gamma = 1.0):
        # invoke the parent (Material) class
        Material.__init__(self, rho, E, R=R)
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
        self.C = self.R
        # the mode-mixity parameter
        self.alpha = 1.0

    # Function to compute the effective unit tangent ("normal to the cohesive boundary") at an interface
    def compute_effective_unit_tangent(self, rp_left_interface, rp_right_interface):
        average_rp_interface = (rp_left_interface + rp_right_interface)/2.0
        effective_unit_tangent_interface = average_rp_interface/np.linalg.norm(average_rp_interface, ord=2, axis=0, keepdims=True)
        return effective_unit_tangent_interface
    
    # Function to compute effective force at an interface
    def compute_effective_force(self, rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, moments_left_interface, moments_right_interface):
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        average_forces_interface = (forces_left_interface + forces_right_interface)/2.0
        average_axial_force_interface = np.sum(average_forces_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial forces are tensile
        average_tensile_force_interface = np.maximum(average_axial_force_interface, 0.0)
        average_bending_moments_interface = (moments_left_interface + moments_right_interface)/2.0
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
    def compute_effective_separation(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI=np.zeros([3,1]), tangent_jumps_DI=np.zeros([3,1])):
        # only if the axial jump is tensile
        tensile_jump = np.maximum(self.compute_axial_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI), 0.0)
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface - tangent_jumps_DI
        tangent_jumps_L2 = np.linalg.norm(rp_jump_interface, ord=2, axis=0, keepdims=True)
        # considering mixed-mode fracture with tensile jumps and bending rotations
        delta = np.sqrt((tensile_jump**2.0)+((self.alpha*self.C*tangent_jumps_L2)**2.0))
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
    
    # Function to compute the cohesive axial forces at the interface
    def compute_cohesive_axial_forces(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3,1]), tangent_jumps_DI=np.zeros([3,1])):
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI, tangent_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # only if the axial jump is tensile
        tensile_jump = np.maximum(self.compute_axial_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI), 0.0)
        # compute the cohesive axial forces
        cohesive_axial_forces = (fcoh/delta)*tensile_jump*effective_unit_tangent_interface
        return cohesive_axial_forces

    # Function to compute the cohesive bending moments at the interface
    def compute_cohesive_bending_moments(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3,1]), tangent_jumps_DI=np.zeros([3,1])):
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI, tangent_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface - tangent_jumps_DI
        # compute the cohesive bending moments
        cohesive_bending_moments = (fcoh/delta)*((self.alpha*self.C)**2.0)*rp_jump_interface
        return cohesive_bending_moments
    
    # Function to compute the derivative of the cohesive bending moments perpendicular to cohesive boundary at the interface
    def compute_cohesive_bending_moments_perpendicular_derivative(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, rpp_left_interface, rpp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3,1]), tangent_jumps_DI=np.zeros([3,1])):
        # effective separation at the interface
        delta = self.compute_effective_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI, tangent_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(delta, delta_max, effective_force_DI)
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface - position_jumps_DI
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface - tangent_jumps_DI
        tangeff_dyd_tangeff = np.matmul(effective_unit_tangent_interface, np.transpose(effective_unit_tangent_interface))
        # bending rotations at the interface
        bending_rotations = np.matmul((np.eye(3)-tangeff_dyd_tangeff), rp_jump_interface)
        # the derivative terms
        # term-1
        dmoments_drots = (fcoh/delta)*((self.alpha*self.C)**2.0)*np.eye(3)
        # term-2
        dfcoh_ddelta = self.compute_effective_cohesive_force_derivative(delta, delta_max, effective_force_DI)
        dmoments_ddelta = ((dfcoh_ddelta/delta)-(fcoh/(delta**2.0)))*((self.alpha*self.C)**2.0)*bending_rotations
        # term-3
        tangent_left_interface = rp_left_interface/np.linalg.norm(rp_left_interface, ord=2, axis=0, keepdims=True)
        tangent_right_interface = rp_right_interface/np.linalg.norm(rp_right_interface, ord=2, axis=0, keepdims=True)
        average_tangent_interface = (tangent_left_interface + tangent_right_interface)/2.0
        average_tangent_interface_L2 = np.linalg.norm(average_tangent_interface, ord=2, axis=0, keepdims=True)
        avtan_dyd_avtan = np.matmul(average_tangent_interface, np.transpose(average_tangent_interface))
        dtangeffdds_coeff = (np.eye(3)/average_tangent_interface_L2)-(avtan_dyd_avtan/(average_tangent_interface_L2**3.0))
        rpdydrp_left_interface = np.matmul(rp_left_interface, np.transpose(rp_left_interface))
        rpdydrp_right_interface = np.matmul(rp_right_interface, np.transpose(rp_right_interface))
        rp_left_interface_L2 = np.linalg.norm(rp_left_interface, ord=2, axis=0, keepdims=True)
        rp_right_interface_L2 = np.linalg.norm(rp_right_interface, ord=2, axis=0, keepdims=True)
        dtangds_left_interface = np.matmul(((np.eye(3)/rp_left_interface_L2)-(rpdydrp_left_interface/(rp_left_interface_L2**3.0))), rpp_left_interface)
        dtangds_right_interface = np.matmul(((np.eye(3)/rp_right_interface_L2)-(rpdydrp_right_interface/(rp_right_interface_L2**3.0))), rpp_right_interface)
        dtangeffds_interface = np.matmul(dtangeffdds_coeff, (dtangds_left_interface+dtangds_right_interface)/2.0)
        rpp_jump_interface = rpp_left_interface - rpp_right_interface
        dbendrots_ds_term1 = np.matmul((np.eye(3)-tangeff_dyd_tangeff), rpp_jump_interface)
        dbendrots_ds_term2 = np.sum(rp_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)*dtangeffds_interface
        dbendrots_ds_term3 = np.sum(rp_jump_interface*dtangeffds_interface, axis=0, keepdims=True)*effective_unit_tangent_interface
        dbendrots_ds = dbendrots_ds_term1 - dbendrots_ds_term2 - dbendrots_ds_term3
        # term-4
        bendrots_dot_dbendrots_ds = np.sum(bending_rotations*dbendrots_ds, axis=0, keepdims=True)
        tensile_jump = np.maximum(self.compute_axial_separation(r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI), 0.0)
        ddelta_ds_term1 = (tensile_jump*np.heaviside(tensile_jump, 0.0)*(np.sum(r_jump_interface*dtangeffds_interface, axis=0, keepdims=True) + np.sum(rp_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)))/delta
        ddelta_ds_term2 = (((self.alpha*self.C)**2.0)*bendrots_dot_dbendrots_ds)/delta
        ddelta_ds = ddelta_ds_term1 + ddelta_ds_term2
        cohesive_bending_moments_perpendicular_derivative = (dmoments_ddelta*ddelta_ds) + np.matmul(dmoments_drots, dbendrots_ds)
        return cohesive_bending_moments_perpendicular_derivative

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
    
    # Function to compute the coefficients of the cohesive force derivatives - IMPLEMENT LATER!!!
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
        