from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
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

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # apply the boundary conditions
    # clamp left node
    bctypes[0, 0:6] = 1
    bcvalues[0, 3] = 1
    # apply constant unit force on right most node  
    bcvalues[1, 0] = 1.0

    # set the boundary conditions
    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(10, 1.0E-05)

    print(solver.solution)

# run main function
main()