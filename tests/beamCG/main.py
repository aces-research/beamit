from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
import numpy as np

def static_main():

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
    Nel = 2
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

    # clamp left node (a clamp fixes both the position and the direction of the tangent)
    # Note that the tangent vector r' is a unit vector if and only if the axial strain is zero.
    # Therefore imposing that the tangent vector is [1, 0, 0] implies that both 
    # 1) the tangent is horizontal, and 
    # 2) the strain at the clamped end is zero
    # However, in general we want to only impose condition 1) as we do not know the strain at the 
    # clamped end a priori. Therefore imposing a horizontal tangent (e.g. aligned with the x-axis) 
    # is achieved by imposing that the y and z component of the tangent be fixed and equal to 0,
    # whereas the x component of the tangent is free (zero Neumann boundary condition). 
    # See also Meier 2014, CMAME for more details.  
    bctypes[0, 0:3] = 1
    bctypes[0, 4:6] = 1
    # apply constant unit force on right most node  
    bcvalues[Nel, 0] = 1.0

    # apply the boundary conditions
    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(10, 1.0E-05)

    print(solver.solution)

def dynamic_main():

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
    nodal_coordinates = function_space.nodes

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewmarkSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp left node (a clamp fixes both the position and the direction of the tangent)
        if ((x_coord == 0.0) and (y_coord == 0.0) and (z_coord == 0.0)):
            bctypes[i, 0:3] = 1
            bctypes[i, 4:6] = 1
            bcvalues[i, 0:3] = system.state[i, 0:3]
            bcvalues[i, 4:6] = system.state[i, 4:6]

    # solve the nonlinear dynamic problem and update the nodal position in system 
    time_steps = 10000 # No. of time steps
    for i in range(0, time_steps):
        # apply constant unit force on right most node
        bcvalues[Nel, 0] = (i+1)/time_steps
        # apply the boundary conditions
        solver.set_boundary_conditions(bctypes, bcvalues)
        solver.solve(dt = 0.01)

    print(solver.solution)
    print(solver.velocity)
    print(solver.acceleration)

# run main functions
static_main()
dynamic_main()