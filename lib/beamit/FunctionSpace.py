import numpy as np
import sys

class FunctionSpace:

    def __init__(self, s0, s1, E, discretization_type = "CG"):
        # the coordinate of the left end of the beam
        self.s0 = s0
        # the coordinate of the right end of the beam
        self.s1 = s1
        # the number of elements
        self.E = E
        # the discretization type (CG (Continuous Galerkin) or DG (Discontinuous Galerkin))
        if ((discretization_type == "CG") or (discretization_type == "DG")):
            self.discretization_type = discretization_type
        else:
            sys.exit("\nUnknown discretization type in the function space.")
        # the number of degrees of freedom per node (3 positions, 3 rotations)
        self.dof = 6
        # the number of dimensions in the problem
        self.dim = 3
        # no. of nodes per element
        self.npel = 2
        # assuming the elements are connected like a simple chain!!!
        # global connectivity (element number -> global dof number), local connectivity (element number -> local dof number)
        if (self.discretization_type == "CG"):
            # number of nodes in the discretization
            self.N = self.E + 1
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            local_dofs = np.arange(0, dofspel, 1, dtype=np.int64)
            self.global_connectivity = np.zeros([self.E, dofspel], dtype=np.int64)
            # not using for now!!!
            self.local_connectivity = np.ones([self.E, dofspel], dtype=np.int64)*local_dofs
            for i in range(0, self.E):
                self.global_connectivity[i:i+1, :] = global_dofs[self.dof*i:(self.dof*i)+dofspel]
        elif (self.discretization_type == "DG"):
            # number of nodes in the discretization
            self.N = self.E*self.npel
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            local_dofs = np.arange(0, dofspel, 1, dtype=np.int64)
            self.global_connectivity = np.zeros([self.E, dofspel], dtype=np.int64)
            # not using for now!!!
            self.local_connectivity = np.ones([self.E, dofspel], dtype=np.int64)*local_dofs
            for i in range(0, self.E):
                self.global_connectivity[i:i+1, :] = global_dofs[dofspel*i:(dofspel*i)+dofspel]
        else:
            sys.exit("\nConnectivity cannot be generated for the discretization type.")
        # the discretization nodes of the beam
        self.nodes = np.zeros([self.N, self.dim])
        # assuming the elements are of equal length!!!
        self.elL = (self.s1 - self.s0)/self.E
        # the number of quadrature points (Gauss quadrature, degree of exactness = 6)
        self.Q = 4
        # the number of shape functions per element (2 nodal displacements, 2 nodal tangents)
        self.S = 4
        # the shape functions evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_functions = np.zeros([self.Q, self.dim, self.npel*self.dof])
        # the shape function first gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_first_gradients = np.zeros([self.Q, self.dim, self.npel*self.dof])
        # the shape function second gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_second_gradients = np.zeros([self.Q, self.dim, self.npel*self.dof])
        # the jacobian of the transformation from parent to reference configuration (xi -> s)
        self.jacobian = (self.s1 - self.s0)/(2.0*self.E)
        # the integration jacobian x weight for quadrature points (size = (integration points, total dofs, 1))
        self.JxW = np.ones([self.Q, self.npel*self.dof, 1])

    # Function to compute shape functions and its gradients of the element at any point
    def compute_shapes(self, xi):
        # Hermite shape functions and its gradients on the reference element (xi in [-1.0, 1.0])
        Nd1 = 0.25*(2.0 + xi)*((1.0 - xi)**2.0)
        Nt1 = 0.25*(1.0 + xi)*((1.0 - xi)**2.0)
        Nd2 = 0.25*(2.0 - xi)*((1.0 + xi)**2.0)
        Nt2 = -0.25*(1.0 - xi)*((1.0 + xi)**2.0)
        Nd1_xi = 0.25*((1.0 - xi)**2.0) - 0.50*(2.0 + xi)*(1.0 - xi)
        Nt1_xi = 0.25*((1.0 - xi)**2.0) - 0.50*(1.0 + xi)*(1.0 - xi)
        Nd2_xi = -0.25*((1.0 + xi)**2.0) + 0.50*(2.0 - xi)*(1.0 + xi)
        Nt2_xi = 0.25*((1.0 + xi)**2.0) - 0.50*(1.0 - xi)*(1.0 + xi)
        Nd1_xixi = -(1.0 - xi) + 0.50*(2.0 + xi)
        Nt1_xixi = -(1.0 - xi) + 0.50*(1.0 + xi)
        Nd2_xixi = -(1.0 + xi) + 0.50*(2.0 - xi)
        Nt2_xixi = (1.0 + xi) - 0.50*(1.0 - xi)
        Nd1_xixixi = 1.50
        Nt1_xixixi = 1.50
        Nd2_xixixi = -1.50
        Nt2_xixixi = 1.50

        L = self.elL # length of the elements

        # elemental shape function and gradient matrices
        shape_functions = np.array([[Nd1, 0.0, 0.0, 0.50*L*Nt1, 0.0, 0.0, Nd2, 0.0, 0.0, 0.50*L*Nt2, 0.0, 0.0], \
                                    [0.0, Nd1, 0.0, 0.0, 0.50*L*Nt1, 0.0, 0.0, Nd2, 0.0, 0.0, 0.50*L*Nt2, 0.0], \
                                    [0.0, 0.0, Nd1, 0.0, 0.0, 0.50*L*Nt1, 0.0, 0.0, Nd2, 0.0, 0.0, 0.50*L*Nt2]])

        shape_first_gradients = np.array([[Nd1_xi, 0.0, 0.0, 0.50*L*Nt1_xi, 0.0, 0.0, Nd2_xi, 0.0, 0.0, 0.50*L*Nt2_xi, 0.0, 0.0], \
                                    [0.0, Nd1_xi, 0.0, 0.0, 0.50*L*Nt1_xi, 0.0, 0.0, Nd2_xi, 0.0, 0.0, 0.50*L*Nt2_xi, 0.0], \
                                    [0.0, 0.0, Nd1_xi, 0.0, 0.0, 0.50*L*Nt1_xi, 0.0, 0.0, Nd2_xi, 0.0, 0.0, 0.50*L*Nt2_xi]])

        shape_second_gradients = np.array([[Nd1_xixi, 0.0, 0.0, 0.50*L*Nt1_xixi, 0.0, 0.0, Nd2_xixi, 0.0, 0.0, 0.50*L*Nt2_xixi, 0.0, 0.0], \
                                    [0.0, Nd1_xixi, 0.0, 0.0, 0.50*L*Nt1_xixi, 0.0, 0.0, Nd2_xixi, 0.0, 0.0, 0.50*L*Nt2_xixi, 0.0], \
                                    [0.0, 0.0, Nd1_xixi, 0.0, 0.0, 0.50*L*Nt1_xixi, 0.0, 0.0, Nd2_xixi, 0.0, 0.0, 0.50*L*Nt2_xixi]])

        shape_third_gradients = np.array([[Nd1_xixixi, 0.0, 0.0, 0.50*L*Nt1_xixixi, 0.0, 0.0, Nd2_xixixi, 0.0, 0.0, 0.50*L*Nt2_xixixi, 0.0, 0.0], \
                                    [0.0, Nd1_xixixi, 0.0, 0.0, 0.50*L*Nt1_xixixi, 0.0, 0.0, Nd2_xixixi, 0.0, 0.0, 0.50*L*Nt2_xixixi, 0.0], \
                                    [0.0, 0.0, Nd1_xixixi, 0.0, 0.0, 0.50*L*Nt1_xixixi, 0.0, 0.0, Nd2_xixixi, 0.0, 0.0, 0.50*L*Nt2_xixixi]])
        
        return shape_functions, shape_first_gradients, shape_second_gradients, shape_third_gradients

    def discretize(self):
        # subdivision of domain (reference configuration)
        # assuming the "initially straight" beam is along the x-direction!!!
        self.nodes[0:1, 0:1] = self.s0
        self.nodes[self.N-1:self.N, 0:1] = self.s1
        if (self.discretization_type == "CG"):
            for i in range(1, self.N-1):
                self.nodes[i:i+1, 0:1] = self.nodes[i-1:i, 0:1] + self.elL
        elif (self.discretization_type == "DG"):
            for i in range(1, self.N-1, 2):
                self.nodes[i:i+1, 0:1] = self.nodes[i-1:i, 0:1] + self.elL
                self.nodes[i+1:i+2, 0:1] = self.nodes[i:i+1, 0:1]

        # quadrature rule on reference element
        integration_points, integration_weights = np.polynomial.legendre.leggauss(self.Q)

        # evaluate shape functions, their gradients and weights at the quadrature points
        for i in range(0, self.Q):
            el_shape_functions, el_shape_first_gradients, el_shape_second_gradients, _ = \
                                    self.compute_shapes(integration_points[i])
            self.shape_functions[i:i+1, :, :] = el_shape_functions
            self.shape_first_gradients[i:i+1, :, :] = el_shape_first_gradients
            self.shape_second_gradients[i:i+1, :, :] = el_shape_second_gradients
            self.JxW[i:i+1, :, :] *= self.jacobian*integration_weights[i]
        print("\nGenerated the function space.")
        pass