from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
import numpy as np

def test_clamp():

    # density of the material
    rho = 1
    # elastic modulus of beam
    E = 1
    # area of cross section
    A = 1
    # area moment of inertia
    I = 1
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

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
    # clamp first node (position zero, tangent along x-axis)
    bctypes[0, 0:6] = 1
    bcvalues[0, 0:6] = 0
    bcvalues[0, 3] = 1

    # set the boundary conditions
    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(1, 1.0E-05)

    # assert that the solution is a straight beam aligned with the x-axis
    assert(np.all(np.transpose(solver.solution) -
        np.array([0.,0.,0.,1.,0.,0.,L,0.,0.,1.,0.,0.])
        ==0))

def test_fix_position():

    # density of the material
    rho = 1
    # elastic modulus of beam
    E = 1
    # area of cross section
    A = 1
    # area moment of inertia
    I = 1
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

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
    # all displacement DOFs fixed to undeformed 
    bctypes[0, 0:3] = 1
    bcvalues[0, 0:3] = 0
    bctypes[1, 0:3] = 1
    bcvalues[1, 0] = L
    bcvalues[1, 1:3] = 0

    # set the boundary conditions
    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(1, 1.0E-05)

    # assert that the solution is a straight beam aligned with the x-axis
    assert(np.all(np.transpose(solver.solution) -
        np.array([0.,0.,0.,1.,0.,0.,L,0.,0.,1.,0.,0.])
        ==0))


def test_fix_tangent():

    # density of the material
    rho = 1
    # elastic modulus of beam
    E = 1
    # area of cross section
    A = 1
    # area moment of inertia
    I = 1
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

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
    # all the tangents are [1,0,0] 
    bctypes[0, 3:6] = 1
    bcvalues[0, 3] = 1
    bctypes[1, 3:6] = 1
    bcvalues[1, 3] = 1

    # fix the right end
    bctypes[1, 0:3] = 1
    bcvalues[1, 0] = L

    # set the boundary conditions
    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(1, 1.0E-05)

    # assert that the solution is a straight beam aligned with the x-axis
    assert(np.all(np.transpose(solver.solution) -
        np.array([0.,0.,0.,1.,0.,0.,L,0.,0.,1.,0.,0.])
        ==0))