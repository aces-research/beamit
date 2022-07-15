from lib.sgi import FunctionSpace
from lib.sgi import Material
from lib.sgi import System
from lib.sgi import WeakForm
from lib.sgi import Solver
import numpy as np

def main():

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
    # all DOFs fixed on the left most node
    solver.bctypes[0:1, 0:6] = 0
    # constant unit force applied on the right most node
    solver.bcvalues[1:2, 0:1] = 1.0
    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(10)

# run main function
main()