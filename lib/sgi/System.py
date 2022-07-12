import numpy as np
from sgi import Point 

class System:
    
    def __init__(self, function_space, material):
        # the geometrical information
        self.function_space = function_space
        # the physical information
        self.material = material
        # the current state of the system
        self.nodal_coordinates = np.empty(function_space.N, dtype=Point.Point)