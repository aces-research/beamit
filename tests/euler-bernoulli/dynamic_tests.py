from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
import numpy as np
import copy

# density of the material
rho = 7850.0
# elastic modulus of beam
E = 2.0E11
# the radius of the beam
R = 1.0E-03
# length of beam
L = 0.10
# number of elements
Nel = 4

# applied loads and tolerances
LOAD = -1.0e01
SPATIAL_TOLERANCE = 1.0E-05
NUMERICAL_TOLERANCE = 1.0E-04

# time steps
dt = 1.0E-06
time_steps = 10000

def test_simply_supported():

    # physical information (material parameters)
    material = Material.Material(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    # the solver
    solver = Solver.ExplicitNewmarkSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # pin at left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
        # roller at the right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bctypes[i, 1] = 1
    
    solver.set_boundary_conditions(bctypes, bcvalues)

    # set initial conditions
    initial_velocity = np.zeros([function_space.N, function_space.dof])
    solver.set_initial_conditions(initial_state, initial_velocity)

    # solve the problem
    for i in range(0, time_steps):
        print("\nCurrent time step =", i+1,"out of", time_steps, "time steps.")
        load_level = (i+1) / time_steps
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            # transverse load at the center
            if (abs(x_coord - 0.50 * L) <= SPATIAL_TOLERANCE):
                bcvalues[n, 1] = 0.50 * load_level * LOAD
            solver.modify_boundary_condition_values(bcvalues)
            # solve the dynamic problem
            solver.solve(dt)

    # test the displacements
    for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[n, 0]
        if (x_coord <= 0.50 * L):
            shifted_coordinate = x_coord
        else:
            shifted_coordinate = L - x_coord
        analytical_transverse_displacement = (LOAD*shifted_coordinate*((3.0*L*L)-(4.0*shifted_coordinate*shifted_coordinate)))/(48.0*E*material.I)
        assert abs(system.state[n, 1] - analytical_transverse_displacement) < NUMERICAL_TOLERANCE, \
               f"Error in X displacement: expected {analytical_transverse_displacement} but computed {system.state[n, 1]}."

def test_bar_tension():

    # physical information (material parameters)
    material = Material.Material(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    # the solver
    solver = Solver.ExplicitNewmarkSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # fixity at the left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
    
    solver.set_boundary_conditions(bctypes, bcvalues)

    # set initial conditions
    initial_velocity = np.zeros([function_space.N, function_space.dof])
    solver.set_initial_conditions(initial_state, initial_velocity)

    # solve the problem
    for i in range(0, time_steps):
        print("\nCurrent time step =", i+1,"out of", time_steps, "time steps.")
        load_level = (i+1) / time_steps
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            # axial load at the right end
            if (abs(x_coord - L) <= SPATIAL_TOLERANCE):
                bcvalues[n, 0] = -load_level * LOAD
            solver.modify_boundary_condition_values(bcvalues)
            # solve the dynamic problem
            solver.solve(dt)

    # test the displacements
    for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[n, 0]
        analytical_x_displacement = (-LOAD*x_coord)/(E*material.A)
        assert abs(system.state[n, 0] - analytical_x_displacement) < NUMERICAL_TOLERANCE*1.0E-05, \
               f"Error in X displacement: expected {analytical_x_displacement} but computed {system.state[n, 0]}."

if __name__ == "__main__":

    # run the tests
    test_simply_supported()
    test_bar_tension()