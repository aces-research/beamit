
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