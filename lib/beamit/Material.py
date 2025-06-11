import numpy as np

class ShearFlexibleMaterial:

    def __init__(self, rho, E, nu, A, I, I_minor):
        """
        Initialize Material.

        Note: Only rectangular and circular beam cross sections are supported in this class.

        Parameters:
            rho: Density of the material
            E: Elastic modulus of the material
            nu: Poisson's ratio of the material
            A: Area of the beam cross section
            I: Area moment of inertia about the major axis
            I_minor: Area moment of inertia about the minor axis
        """
        self.rho = rho
        self.E = E
        self.A = A
        self.I = I
        self.I_minor = I_minor
        # the shear modulus
        self.G = E/(2.0*(1.0 + nu))
        # the reduced area of cross section (same for both rectangular and circular cross sections)
        self.A_red = (5/6)*A
        # the torsional moment of inertia
        self.I_T = self.I + self.I_minor  # works for symmetrical cross-sections

class TFKLMaterial(ShearFlexibleMaterial):

    def __init__(self, rho, E, R):
        """
        Initialize TFKLMaterial.

        Note: Only circular beam cross sections are supported in this class.

        Parameters:
            rho: Density of the material
            E: Elastic modulus of the material
            R: Beam radius
        """
        # initialize the parent (ShearFlexibleMaterial) class
        ShearFlexibleMaterial.__init__(
            self, rho, E, nu=0.0, A=np.pi*(R**2.0), I=(np.pi*(R**4.0))/4.0, I_minor=(np.pi*(R**4.0))/4.0)
        self.rho = rho
        self.E = E
        self.R = R
        self.G = None # shear modulus is not defined for TFKLMaterial
        print("\nCreated the material.")

class TFKLCohesiveInterfaceMaterial(TFKLMaterial):

    def __init__(self, rho, E, R, Sc, Gc, alpha=1.0):
        """
        Initialize TFKLCohesiveInterfaceMaterial.

        Parameters:
            rho: Density of the material
            E: Elastic modulus
            R: Beam radius
            Sc: Critical effective cohesive strength of the material
            Gc: Effective fracture energy of the material
            alpha: Mode-mixity parameter (default is 1.0)
        """
        # invoke the parent (TFKLMaterial) class
        TFKLMaterial.__init__(self, rho, E, R=R)
        self.Sc = Sc
        self.Gc = Gc
        self.delta_c = (2.0*self.Gc)/self.Sc
        self.fc = self.Sc*self.A
        self.alpha = alpha

    def compute_effective_unit_tangent(self, rp_left_interface, rp_right_interface):
        """
        Compute the effective unit tangent ("normal to the cohesive boundary") at an interface.
        Parameters:
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
        Returns:
            effective_unit_tangent_interface: Effective unit tangent vector at the interface
        """
        average_rp_interface = (rp_left_interface + rp_right_interface)/2.0
        effective_unit_tangent_interface = average_rp_interface / \
            np.linalg.norm(average_rp_interface, ord=2, axis=0, keepdims=True)
        return effective_unit_tangent_interface

    def compute_effective_force(self, rp_left_interface, rp_right_interface, forces_left_interface, 
                                forces_right_interface, moments_left_interface, 
                                moments_right_interface):
        """
        Compute the effective force at an interface.
        Parameters:
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
            forces_left_interface: Forces at the left side of the interface
            forces_right_interface: Forces at the right side of the interface
            moments_left_interface: Bending moments at the left side of the interface
            moments_right_interface: Bending moments at the right side of the interface
        Returns:
            effective_force_interface: Effective force at the interface
        """
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(
            rp_left_interface, rp_right_interface)
        average_forces_interface = (
            forces_left_interface + forces_right_interface)/2.0
        average_axial_force_interface = np.sum(
            average_forces_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        # only if the axial forces are tensile
        average_tensile_force_interface = np.maximum(
            average_axial_force_interface, 0.0)
        average_bending_moments_interface = (
            moments_left_interface + moments_right_interface)/2.0
        average_bending_moments_interface_L2 = np.linalg.norm(
            average_bending_moments_interface, ord=2, axis=0, keepdims=True)
        # considering mixed-mode fracture with tensile forces and bending moments
        effective_force_interface = np.sqrt((average_tensile_force_interface**2.0)+(
            (average_bending_moments_interface_L2/(self.alpha*self.R))**2.0))
        return effective_force_interface

    def evaluate_damage_initiation_criterion(self, rp_left_interface, rp_right_interface, 
                                             forces_left_interface, forces_right_interface, 
                                             moments_left_interface, moments_right_interface):
        """
        Evaluate the damage initiation criterion at an interface.
        Parameters:
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
            forces_left_interface: Forces at the left side of the interface
            forces_right_interface: Forces at the right side of the interface
            moments_left_interface: Bending moments at the left side of the interface
            moments_right_interface: Bending moments at the right side of the interface
        Returns:
            True if the damage initiation criterion is satisfied, False otherwise.
        """
        effective_force_interface = self.compute_effective_force(
            rp_left_interface, rp_right_interface, forces_left_interface, forces_right_interface, 
            moments_left_interface, moments_right_interface)
        # if the effective force at the interface satisfies the damage initiation criterion
        if (effective_force_interface/self.fc >= 1.0):
            return True
        else:
            return False

    def compute_effective_separation(self, r_left_interface, r_right_interface, rp_left_interface, 
                                     rp_right_interface, position_jumps_DI=np.zeros([3, 1]), 
                                     tangent_jumps_DI=np.zeros([3, 1])):
        """
        Compute the effective separation at the interface (across the "cohesive boundary").
        Parameters:
            r_left_interface: Position vector at the left side of the interface
            r_right_interface: Position vector at the right side of the interface
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
            position_jumps_DI: Position jumps at damage initiation (default is zero)
            tangent_jumps_DI: Tangent jumps at damage initiation (default is zero)
        Returns:
            delta: Effective separation at the interface
        """
        # only if the axial jump is tensile
        tensile_jump = np.maximum(self.compute_axial_separation(
            r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, 
            position_jumps_DI), 0.0)
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface - tangent_jumps_DI
        tangent_jumps_L2 = np.linalg.norm(
            rp_jump_interface, ord=2, axis=0, keepdims=True)
        # considering mixed-mode fracture with tensile jumps and bending rotations
        delta = np.sqrt((tensile_jump**2.0) +
                        ((self.alpha*self.R*tangent_jumps_L2)**2.0))
        return delta

    def compute_axial_separation(self, r_left_interface, r_right_interface, rp_left_interface, 
                                 rp_right_interface, position_jumps_DI=np.zeros([3, 1])):
        """
        Compute the axial separation at the interface.
        Parameters:
            r_left_interface: Position vector at the left side of the interface
            r_right_interface: Position vector at the right side of the interface
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
            position_jumps_DI: Position jumps at damage initiation (default is zero)
        Returns:
            axial_jump: Axial separation at the interface
        """
        # compute the effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(
            rp_left_interface, rp_right_interface)
        # position jump at the interface
        r_jump_interface = r_right_interface - r_left_interface - position_jumps_DI
        axial_jump = np.sum(
            r_jump_interface*effective_unit_tangent_interface, axis=0, keepdims=True)
        return axial_jump

    def compute_effective_maximum_separation(self, delta, delta_max):
        """
        Compute the "new" (updated) maximum effective separation at the interface.
        Parameters:
            delta: Effective separation at the interface
            delta_max: Current maximum effective separation at the interface
        Returns:
            new_delta_max: New (updated) maximum effective separation at the interface
        """
        if (delta >= self.delta_c): # complete damage
            new_delta_max = self.delta_c
        elif (delta >= delta_max): # loading
            new_delta_max = delta
        else: # unloading
            new_delta_max = delta_max
        return new_delta_max

    def compute_effective_cohesive_force(self, delta, delta_max, effective_force_DI=None):
        """
        Compute the effective cohesive force at the interface according to a linear TSL.
        Parameters:
            delta: Effective separation at the interface
            delta_max: Current maximum effective separation at the interface
            effective_force_DI: Effective cohesive force at damage initiation (default is None, which means it will be set to critical force)
        Returns:
            fcoh: Effective cohesive force at the interface
        """
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

    def compute_cohesive_axial_forces(self, r_left_interface, r_right_interface, rp_left_interface, 
                                      rp_right_interface, delta_max, effective_force_DI=None, 
                                      position_jumps_DI=np.zeros([3, 1]), 
                                      tangent_jumps_DI=np.zeros([3, 1])):
        """
        Compute the cohesive axial forces at the interface.
        Parameters:
            r_left_interface: Position vector at the left side of the interface
            r_right_interface: Position vector at the right side of the interface
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
            delta_max: Current maximum effective separation at the interface
            effective_force_DI: Effective cohesive force at damage initiation (default is None, which means it will be set to critical force)
            position_jumps_DI: Position jumps at damage initiation (default is zero)
            tangent_jumps_DI: Tangent jumps at damage initiation (default is zero)
        Returns:
            cohesive_axial_forces: Cohesive axial forces at the interface
        """
        # effective separation at the interface
        delta = self.compute_effective_separation(
            r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI, tangent_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(
            delta, delta_max, effective_force_DI)
        # effective unit tangent at the interface
        effective_unit_tangent_interface = self.compute_effective_unit_tangent(
            rp_left_interface, rp_right_interface)
        # only if the axial jump is tensile
        tensile_jump = np.maximum(self.compute_axial_separation(
            r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI), 0.0)
        # compute the cohesive axial forces
        cohesive_axial_forces = (fcoh/delta)*tensile_jump * \
            effective_unit_tangent_interface
        return cohesive_axial_forces

    def compute_cohesive_bending_moments(self, r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, delta_max, effective_force_DI=None, position_jumps_DI=np.zeros([3, 1]), tangent_jumps_DI=np.zeros([3, 1])):
        """
        Compute the cohesive bending moments at the interface.
        Parameters:
            r_left_interface: Position vector at the left side of the interface
            r_right_interface: Position vector at the right side of the interface
            rp_left_interface: Tangent vector at the left side of the interface
            rp_right_interface: Tangent vector at the right side of the interface
            delta_max: Current maximum effective separation at the interface
            effective_force_DI: Effective cohesive force at damage initiation (default is None, which means it will be set to critical force)
            position_jumps_DI: Position jumps at damage initiation (default is zero)
            tangent_jumps_DI: Tangent jumps at damage initiation (default is zero)
        Returns:
            cohesive_bending_moments: Cohesive bending moments at the interface
        """
        # effective separation at the interface
        delta = self.compute_effective_separation(
            r_left_interface, r_right_interface, rp_left_interface, rp_right_interface, position_jumps_DI, tangent_jumps_DI)
        # effective cohesive force at the interface
        fcoh = self.compute_effective_cohesive_force(
            delta, delta_max, effective_force_DI)
        # tangent jump at the interface
        rp_jump_interface = rp_right_interface - rp_left_interface - tangent_jumps_DI
        # compute the cohesive bending moments
        cohesive_bending_moments = (
            fcoh/delta)*((self.alpha*self.R)**2.0)*rp_jump_interface
        return cohesive_bending_moments

    def compute_effective_cohesive_force_derivative(self, delta, delta_max, effective_force_DI=None):
        """
        Compute the derivative of the effective cohesive force w.r.t the effective separation.
        Parameters:
            delta: Effective separation at the interface
            delta_max: Current maximum effective separation at the interface
            effective_force_DI: Effective cohesive force at damage initiation (default is None, which means it will be set to critical force)
        Returns:
            dfcoh_ddelta: Derivative of the effective cohesive force w.r.t the effective separation
        """
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
