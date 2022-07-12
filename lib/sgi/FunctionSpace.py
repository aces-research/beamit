import numpy as np
from sgi import Point 

class FunctionSpace:

    def __init__(self, s0, s1, E):
        # the coordinate of the left end of the beam
        self.s0 = s0
        # the coordinate of the right end of the beam
        self.s1 = s1
        # the number of elements
        self.E = E
        # the number of nodes
        self.N = E + 1
        # the number of degrees of freedom per node (3 positions, 3 rotations)
        self.dof = 6
        # the discretization nodes of the beam
        self.nodes = np.empty(self.N, dtype=Point.Point)
        # the number of quadrature points (Gauss quadrature, degree of exactness 4)
        self.Q = 3
        # the number of shape functions per element (2 nodal displacements, 2 nodal tangents)
        self.S = 4
        # the shape functions evaluated at quadrature points
        self.shape_functions = np.empty([self.E, self.Q, self.S])

    def discretize(self):

        # build subdivision of domain, populate self.nodes

        # build shape functions on reference element

        # build quadrature rule on reference element

        # evaluate shape functions on quadrature points (store result in self.shape_functions)

        pass