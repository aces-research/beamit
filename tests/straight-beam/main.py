from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
import numpy as np

def test():

    # elastic modulus of beam
    E = 1
    # area of cross section
    A = 1
    # area moment of inertia
    I = 1
    # physical information (material parameters)
    material = Material.Material(E, A, I)

    # length of beam
    L = 1
    # number of elements
    Nel = 1
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "CG")
    function_space.discretize()

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # apply the boundary conditions
    # all displacement DOFs fixed 
    solver.bctypes[0, 0:3] = 1
    solver.bcvalues[0, 0:3] = 0
    solver.bctypes[1, 0:3] = 1
    solver.bcvalues[1, 0:3] = 0

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(1, 1.0E-05)

    # assert that the solution is a straight beam aligned with the x-axis
    assert(np.all(np.transpose(solver.solution) -
        np.array([0.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,0.])
        ==0))