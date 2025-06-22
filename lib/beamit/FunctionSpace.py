from abc import ABC, abstractmethod
import numpy as np
import sys

class FunctionSpace(ABC):

    def __init__(self, s0, s1, E, discretization_type="CG"):
        """
        Initialize the FunctionSpace (abstract) class

        Parameters:
            s0: Coordinate of the left end of the beam
            s1: Coordinate of the right end of the beam
            E: Number of elements
            discretization_type: Discretization type, either "CG" (Continuous Galerkin) or "DG" (Discontinuous Galerkin)
        """
        self.s0 = s0
        self.s1 = s1
        self.E = E
        if ((discretization_type == "CG") or (discretization_type == "DG")):
            self.discretization_type = discretization_type
        else:
            sys.exit("\nUnknown discretization type in the function space.")
        # the number of degrees of freedom per node
        self.dof = None
        # the number of dimensions in the problem
        self.dim = None
        # no. of nodes per element
        self.npel = None
        # global connectivity (element number -> global dof number)
        self.global_connectivity = None
        # local translational dof numbers
        self.local_translational_dofs = None
        # local rotational dof numbers
        self.local_rotational_dofs = None
        # the discretization nodes of the beam
        self.nodes = None
        # length of the elements
        # assuming the elements are of equal length!!!
        self.elL = (self.s1 - self.s0)/self.E
        # the number of quadrature points
        self.Q = None
        # the shape functions evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_functions = None
        # the shape function first gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_first_gradients = None
        # the jacobian of the transformation from parent to reference configuration (xi -> s)
        self.jacobian = None
        # the integration jacobian x weight for quadrature points
        self.JxW = None

    @abstractmethod
    def compute_shapes(self, xi):
        """
        Compute the shape functions and their gradients at a given point xi.

        Parameters:
            xi: The point in the reference element (xi in [-1.0, 1.0])
        Returns:
            shape_functions: The shape functions evaluated at the point xi
            shape_first_gradients: The first gradients of the shape functions evaluated at the point xi
        """
        pass

    @abstractmethod
    def discretize(self):
        """
        Generate the nodal coordinates, and shape functions and their derivatives.
        """
        pass

class TFKLGeometricallyExactFunctionSpace(FunctionSpace):

    def __init__(self, s0, s1, E, discretization_type="CG"):
        """
        Initialize the function space for the torsion-free Kirchhoff-Love Geometrically exact beam.

        Parameters:
            s0: Coordinate of the left end of the beam
            s1: Coordinate of the right end of the beam
            E: Number of elements
            discretization_type: Discretization type, either "CG" (Continuous Galerkin) or "DG" (Discontinuous Galerkin)
        """
        # initialize the parent (FunctionSpace) class
        FunctionSpace.__init__(self, s0, s1, E, discretization_type)
        # the number of degrees of freedom per node (3 positions, 3 rotations)
        self.dof = 6
        # the number of dimensions in the problem
        self.dim = 3
        # no. of nodes per element
        self.npel = 2
        # global connectivity (element number -> global dof number), local connectivity (element number -> local dof number)
        # assuming the elements are connected like a simple chain!!!
        if (self.discretization_type == "CG"):
            # number of nodes in the discretization
            self.N = self.E + 1
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            self.global_connectivity = np.zeros(
                [self.E, dofspel], dtype=np.int64)
            for i in range(0, self.E):
                self.global_connectivity[i:i+1,
                                         :] = global_dofs[self.dof*i:(self.dof*i)+dofspel]
        elif (self.discretization_type == "DG"):
            # number of nodes in the discretization
            self.N = self.E*self.npel
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            self.global_connectivity = np.zeros(
                [self.E, dofspel], dtype=np.int64)
            for i in range(0, self.E):
                self.global_connectivity[i:i+1,
                                         :] = global_dofs[dofspel*i:(dofspel*i)+dofspel]
        # local translational dof numbers
        self.local_translational_dofs = np.array([0, 1, 2, 6, 7, 8], dtype=np.int64)
        # local rotational dof numbers
        self.local_rotational_dofs = np.array([3, 4, 5, 9, 10, 11], dtype=np.int64)
        # the discretization nodes of the beam
        self.nodes = np.zeros([self.N, self.dim])
        # the number of quadrature points (Gauss quadrature, degree of exactness = 6)
        self.Q = 4
        # the shape functions evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_functions = np.zeros([self.Q, self.dim, self.npel*self.dof])
        # the shape function first gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_first_gradients = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the shape function second gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.shape_second_gradients = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the jacobian of the transformation from parent to reference configuration (xi -> s)
        self.jacobian = self.elL / 2.0
        # the integration jacobian x weight for quadrature points (size = (integration points, total dofs, 1))
        self.JxW = np.ones([self.Q, self.npel*self.dof, 1])

    def compute_shapes(self, xi):
        """
        Compute the shape functions and their gradients at a given point xi.

        Parameters:
            xi: The point in the reference element (xi in [-1.0, 1.0])

        Returns:
            shape_functions: The shape functions evaluated at the point xi
            shape_first_gradients: The first gradients of the shape functions evaluated at the point xi
            shape_second_gradients: The second gradients of the shape functions evaluated at the point xi
            shape_third_gradients: The third gradients of the shape functions evaluated at the point xi
        """
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

        L = self.elL  # length of the elements

        # elemental shape function and gradient matrices
        shape_functions = np.array([[Nd1, 0.0, 0.0, 0.50*L*Nt1, 0.0, 0.0, Nd2, 0.0, 0.0, 0.50*L*Nt2, 0.0, 0.0],
                                    [0.0, Nd1, 0.0, 0.0, 0.50*L*Nt1, 0.0,
                                        0.0, Nd2, 0.0, 0.0, 0.50*L*Nt2, 0.0],
                                    [0.0, 0.0, Nd1, 0.0, 0.0, 0.50*L*Nt1, 0.0, 0.0, Nd2, 0.0, 0.0, 0.50*L*Nt2]])

        shape_first_gradients = np.array([[Nd1_xi, 0.0, 0.0, 0.50*L*Nt1_xi, 0.0, 0.0, Nd2_xi, 0.0, 0.0, 0.50*L*Nt2_xi, 0.0, 0.0],
                                          [0.0, Nd1_xi, 0.0, 0.0, 0.50*L*Nt1_xi, 0.0,
                                              0.0, Nd2_xi, 0.0, 0.0, 0.50*L*Nt2_xi, 0.0],
                                          [0.0, 0.0, Nd1_xi, 0.0, 0.0, 0.50*L*Nt1_xi, 0.0, 0.0, Nd2_xi, 0.0, 0.0, 0.50*L*Nt2_xi]])

        shape_second_gradients = np.array([[Nd1_xixi, 0.0, 0.0, 0.50*L*Nt1_xixi, 0.0, 0.0, Nd2_xixi, 0.0, 0.0, 0.50*L*Nt2_xixi, 0.0, 0.0],
                                           [0.0, Nd1_xixi, 0.0, 0.0, 0.50*L*Nt1_xixi, 0.0,
                                               0.0, Nd2_xixi, 0.0, 0.0, 0.50*L*Nt2_xixi, 0.0],
                                           [0.0, 0.0, Nd1_xixi, 0.0, 0.0, 0.50*L*Nt1_xixi, 0.0, 0.0, Nd2_xixi, 0.0, 0.0, 0.50*L*Nt2_xixi]])

        shape_third_gradients = np.array([[Nd1_xixixi, 0.0, 0.0, 0.50*L*Nt1_xixixi, 0.0, 0.0, Nd2_xixixi, 0.0, 0.0, 0.50*L*Nt2_xixixi, 0.0, 0.0],
                                          [0.0, Nd1_xixixi, 0.0, 0.0, 0.50*L*Nt1_xixixi, 0.0,
                                              0.0, Nd2_xixixi, 0.0, 0.0, 0.50*L*Nt2_xixixi, 0.0],
                                          [0.0, 0.0, Nd1_xixixi, 0.0, 0.0, 0.50*L*Nt1_xixixi, 0.0, 0.0, Nd2_xixixi, 0.0, 0.0, 0.50*L*Nt2_xixixi]])

        return shape_functions, shape_first_gradients, shape_second_gradients, shape_third_gradients

    def discretize(self):
        """
        Generate the nodal coordinates, and shape functions and their derivatives.

        Note: We assume that the **initially straight beam** is aligned along the x-axis.
        """
        # subdivision of domain (reference configuration)
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
        integration_points, integration_weights = np.polynomial.legendre.leggauss(
            self.Q)

        # evaluate shape functions, their gradients and weights at the quadrature points
        for i in range(0, self.Q):
            el_shape_functions, el_shape_first_gradients, el_shape_second_gradients, _ = \
                self.compute_shapes(integration_points[i])
            self.shape_functions[i:i+1, :, :] = el_shape_functions
            self.shape_first_gradients[i:i+1, :, :] = el_shape_first_gradients
            self.shape_second_gradients[i:i+1,
                                        :, :] = el_shape_second_gradients
            self.JxW[i:i+1, :, :] *= self.jacobian*integration_weights[i]
        print("\nGenerated the function space.")

class EulerBernoulliFunctionSpace(FunctionSpace):

    def __init__(self, s0, s1, E, discretization_type="CG"):
        """
        Initialize the function space for the Euler-Bernoulli beam.

        Parameters:
            s0: Coordinate of the left end of the beam
            s1: Coordinate of the right end of the beam
            E: Number of elements
            discretization_type: Discretization type, either "CG" (Continuous Galerkin) or "DG" (Discontinuous Galerkin)
        """
        # initialize the parent (FunctionSpace) class
        FunctionSpace.__init__(self, s0, s1, E, discretization_type)
        # the number of degrees of freedom per node (2 displacements, 1 rotation)
        self.dof = 3
        # the number of dimensions in the problem
        self.dim = 1
        # no. of nodes per element
        self.npel = 2
        # global connectivity (element number -> global dof number)
        # assuming the elements are connected like a simple chain!!!
        if (self.discretization_type == "CG"):
            # number of nodes in the discretization
            self.N = self.E + 1
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            self.global_connectivity = np.zeros(
                [self.E, dofspel], dtype=np.int64)
            for i in range(0, self.E):
                self.global_connectivity[i:i+1,
                                         :] = global_dofs[self.dof*i:(self.dof*i)+dofspel]
        elif (self.discretization_type == "DG"):
            # number of nodes in the discretization
            self.N = self.E*self.npel
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            self.global_connectivity = np.zeros(
                [self.E, dofspel], dtype=np.int64)
            for i in range(0, self.E):
                self.global_connectivity[i:i+1,
                                         :] = global_dofs[dofspel*i:(dofspel*i)+dofspel]
        # local translational dof numbers
        self.local_translational_dofs = np.array([0, 1, 3, 4], dtype=np.int64)
        # local rotational dof numbers
        self.local_rotational_dofs = np.array([2, 5], dtype=np.int64)
        # the discretization nodes of the beam
        self.nodes = np.zeros([self.N, self.dim])
        # the number of quadrature points (Gauss quadrature, degree of exactness = 6)
        self.Q = 4
        # the lagrange shape functions evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.lagrange_shape_functions = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the lagrange shape function first gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.lagrange_shape_first_gradients = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the hermite shape functions evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.hermite_shape_functions = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the hermite shape function first gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.hermite_shape_first_gradients = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the hermite shape function second gradients evaluated at quadrature points (size = (integration points, dimensions, total dofs))
        self.hermite_shape_second_gradients = np.zeros(
            [self.Q, self.dim, self.npel*self.dof])
        # the jacobian of the transformation from parent to reference configuration (xi -> s)
        self.jacobian = self.elL / 2.0
        # the integration jacobian x weight for quadrature points (size = (integration points, total dofs, 1))
        self.JxW = np.ones([self.Q, self.npel*self.dof, 1])

    def compute_shapes(self, xi):
        """
        Compute the shape functions and their gradients at a given point xi.
        Parameters:
            xi: The point in the reference element (xi in [-1.0, 1.0])
        Returns:
            shape_functions: The shape functions evaluated at the point xi
            shape_first_gradients: The first gradients of the shape functions evaluated at the point xi
        """
        pass

    def compute_lagrange_shapes(self, xi):
        """
        Compute the lagrange shape functions and their first gradients at a given point xi.

        Parameters:
            xi: The point in the reference element (xi in [-1.0, 1.0])
        Returns:
            shape_functions: The shape functions evaluated at the point xi
            shape_first_gradients: The first gradients of the shape functions evaluated at the point xi
        """
        Nu1 = 0.50*(1.0 - xi)
        Nu2 = 0.50*(1.0 + xi)
        Nu1_xi = -0.50
        Nu2_xi = 0.50
        shape_functions = np.array([[Nu1, 0.0, 0.0, Nu2, 0.0, 0.0]])
        shape_first_gradients = np.array(
            [[Nu1_xi, 0.0, 0.0, Nu2_xi, 0.0, 0.0]])
        return shape_functions, shape_first_gradients

    def compute_hermite_shapes(self, xi):
        """
        Compute the hermite shape functions and their gradients at a given point xi.

        Parameters:
            xi: The point in the reference element (xi in [-1.0, 1.0])
        Returns:
            shape_functions: The shape functions evaluated at the point xi
            shape_first_gradients: The first gradients of the shape functions evaluated at the point xi
            shape_second_gradients: The second gradients of the shape functions evaluated at the point xi
            shape_third_gradients: The third gradients of the shape functions evaluated at the point xi
        """
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

        # elemental shape function and gradient matrices
        L = self.elL
        shape_functions = np.array(
            [[0.0, Nd1, 0.50*L*Nt1, 0.0, Nd2, 0.50*L*Nt2]])
        shape_first_gradients = np.array(
            [[0.0, Nd1_xi, 0.50*L*Nt1_xi, 0.0, Nd2_xi, 0.50*L*Nt2_xi]])
        shape_second_gradients = np.array(
            [[0.0, Nd1_xixi, 0.50*L*Nt1_xixi, 0.0, Nd2_xixi, 0.50*L*Nt2_xixi]])
        shape_third_gradients = np.array(
            [[0.0, Nd1_xixixi, 0.50*L*Nt1_xixixi, 0.0, Nd2_xixixi, 0.50*L*Nt2_xixixi]])

        return shape_functions, shape_first_gradients, shape_second_gradients, shape_third_gradients

    def discretize(self):
        """
        Generate the nodal coordinates, and shape functions and their derivatives.
        """
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
        integration_points, integration_weights = np.polynomial.legendre.leggauss(
            self.Q)

        # evaluate shape functions, their gradients and weights at the quadrature points
        for i in range(0, self.Q):
            el_LShape_functions, el_LShape_first_gradients = \
                self.compute_lagrange_shapes(integration_points[i])
            self.lagrange_shape_functions[i:i+1, :, :] = el_LShape_functions
            self.lagrange_shape_first_gradients[i:i +
                                                1, :, :] = el_LShape_first_gradients
            el_HShape_functions, el_HShape_first_gradients, el_HShape_second_gradients, _ = \
                self.compute_hermite_shapes(integration_points[i])
            self.hermite_shape_functions[i:i+1, :, :] = el_HShape_functions
            self.hermite_shape_first_gradients[i:i +
                                               1, :, :] = el_HShape_first_gradients
            self.hermite_shape_second_gradients[i:i +
                                                1, :, :] = el_HShape_second_gradients
            self.JxW[i:i+1, :, :] *= self.jacobian*integration_weights[i]
        print("\nGenerated the function space.")

class ShearFlexibleGeometricallyExactFunctionSpace(FunctionSpace):

    def __init__(self, s0, s1, E, discretization_type="CG"):
        """
        Initialize the function space for the torsion-free Kirchhoff-Love Geometrically exact beam.

        Parameters:
            s0: Coordinate of the left end of the beam
            s1: Coordinate of the right end of the beam
            E: Number of elements
            discretization_type: Discretization type, either "CG" (Continuous Galerkin) or "DG" (Discontinuous Galerkin)
        """
        # initialize the parent (FunctionSpace) class
        FunctionSpace.__init__(self, s0, s1, E, discretization_type)
        # the number of degrees of freedom per node (3 positions, 3 rotations)
        self.dof = 6
        # the number of dimensions in the problem
        self.dim = 3
        # no. of nodes per element
        self.npel = 2
        # global connectivity (element number -> global dof number)
        # assuming the elements are connected like a simple chain!!!
        if (self.discretization_type == "CG"):
            # number of nodes in the discretization
            self.N = self.E + 1
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            self.global_connectivity = np.zeros(
                [self.E, dofspel], dtype=np.int64)
            for i in range(0, self.E):
                self.global_connectivity[i:i+1,
                                         :] = global_dofs[self.dof*i:(self.dof*i)+dofspel]
        elif (self.discretization_type == "DG"):
            # number of nodes in the discretization
            self.N = self.E*self.npel
            global_dofs = np.arange(0, self.N*self.dof, 1, dtype=np.int64)
            dofspel = self.npel*self.dof
            self.global_connectivity = np.zeros(
                [self.E, dofspel], dtype=np.int64)
            for i in range(0, self.E):
                self.global_connectivity[i:i+1,
                                         :] = global_dofs[dofspel*i:(dofspel*i)+dofspel]
        # local translational dof numbers
        self.local_translational_dofs = np.array([0, 1, 2, 6, 7, 8], dtype=np.int64)
        # local rotational dof numbers
        self.local_rotational_dofs = np.array([3, 4, 5, 9, 10, 11], dtype=np.int64)
        # the discretization nodes of the beam
        self.nodes = np.zeros([self.N, self.dim])
        # the number of quadrature points (Gauss quadrature, degree of exactness = 2)
        # We are using reduced integration to prevent shear locking.
        # This works for integrating the residual and stiffness. But for the mass matrix, we should
        # use the two point Gauss quadrature!!!
        self.Q = 1
        # the shape functions evaluated at quadrature points (size = (integration points, dimensions, translational/rotational dofs))
        self.shape_functions = np.zeros([self.Q, self.dim, self.npel*self.dim])
        # the shape function first gradients evaluated at quadrature points (size = (integration points, dimensions, translational/rotational dofs))
        self.shape_first_gradients = np.zeros(
            [self.Q, self.dim, self.npel*self.dim])
        # the jacobian of the transformation from parent to reference configuration (xi -> s)
        self.jacobian = self.elL / 2.0
        # the integration jacobian x weight for quadrature points (size = (integration points, translational/rotational dofs, 1))
        self.JxW = np.ones([self.Q, self.npel*self.dim, 1])

    def compute_shapes(self, xi):
        """
        Compute the shape functions and their gradients at a given point xi.
        Parameters:
            xi: The point in the reference element (xi in [-1.0, 1.0])
        Returns:
            shape_functions: The shape functions evaluated at the point xi
            shape_first_gradients: The first gradients of the shape functions evaluated at the point xi
        """
        N1 = 0.50*(1.0 - xi)
        N2 = 0.50*(1.0 + xi)
        N1_xi = -0.50
        N2_xi = 0.50
        # elemental shape function and gradient matrices
        shape_functions = np.array([[N1, 0.0, 0.0, N2, 0.0, 0.0],
                                    [0.0, N1, 0.0, 0.0, N2, 0.0],
                                    [0.0, 0.0, N1, 0.0, 0.0, N2]])
        shape_first_gradients = np.array([[N1_xi, 0.0, 0.0, N2_xi, 0.0, 0.0],
                                          [0.0, N1_xi, 0.0, 0.0, N2_xi, 0.0],
                                          [0.0, 0.0, N1_xi, 0.0, 0.0, N2_xi]])
        return shape_functions, shape_first_gradients

    def discretize(self):
        """
        Generate the nodal coordinates, and shape functions and their derivatives.

        Note: We assume that the **initially straight beam** is aligned along the x-axis.
        """
        # subdivision of domain (reference configuration)
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
        integration_points, integration_weights = np.polynomial.legendre.leggauss(
            self.Q)

        # evaluate shape functions, their gradients and weights at the quadrature points
        for i in range(0, self.Q):
            el_shape_functions, el_shape_first_gradients = self.compute_shapes(
                integration_points[i])
            self.shape_functions[i:i+1, :, :] = el_shape_functions
            self.shape_first_gradients[i:i+1, :, :] = el_shape_first_gradients
            self.JxW[i:i+1, :, :] *= self.jacobian*integration_weights[i]
        print("\nGenerated the function space.")
