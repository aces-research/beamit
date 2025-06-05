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
NUMERICAL_TOLERANCE = 1.0E-10

def test_simply_supported():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0.0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

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
        # transverse load at the center
        elif (abs(x_coord - (0.50 * L)) <= SPATIAL_TOLERANCE):
            bcvalues[i, 1] = 0.50 * LOAD

    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the problem
    solver.solve(Nmax=1)

    # test the displacements
    for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[n, 0]
        if (x_coord <= 0.50 * L):
            shifted_coordinate = x_coord
        else:
            shifted_coordinate = L - x_coord
        analytical_transverse_displacement = (LOAD*shifted_coordinate*((3.0*L*L)-(4.0*shifted_coordinate*shifted_coordinate)))/(48.0*E*material.I)
        # asserts
        assert abs(system.state[n, 1] - analytical_transverse_displacement) < NUMERICAL_TOLERANCE, \
               f"Error in X displacement: expected {analytical_transverse_displacement} but computed {system.state[n, 1]}."

def test_cantilever():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0.0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # clamp at left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            bctypes[i, 2] = 1
        # load at the right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bcvalues[i, 1] = LOAD

    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the problem
    solver.solve(Nmax=1)

    # test the displacement
    analytical_displacement = (LOAD*(L**3.0)) / (3.0*E*material.I)
    assert abs(system.state[-1, 1] - analytical_displacement) < NUMERICAL_TOLERANCE, \
               f"Error in X displacement: expected {analytical_displacement} but computed {system.state[-1, 1]}."

def test_bar_tension():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0.0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # fixity at the left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
        # load at the right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bcvalues[i, 0] = -LOAD
    
    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the problem
    solver.solve(Nmax=1)

    # test the displacements
    for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[n, 0]
        analytical_x_displacement = (-LOAD*x_coord)/(E*material.A)
        assert abs(system.state[n, 0] - analytical_x_displacement) < NUMERICAL_TOLERANCE*1.0E-10, \
               f"Error in X displacement: expected {analytical_x_displacement} but computed {system.state[n, 0]}."
        
def test_fixed_fixed():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0.0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # clamp at left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            bctypes[i, 2] = 1
        # clamped at the right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            bctypes[i, 2] = 1
        # transverse load at the center
        elif (abs(x_coord - (0.50 * L)) <= SPATIAL_TOLERANCE):
            bcvalues[i, 1] = 0.50 * LOAD

    solver.set_boundary_conditions(bctypes, bcvalues)

    # solve the problem
    solver.solve(Nmax=1)

    # test the displacements and moments
    analytical_displacement = (LOAD*(L**3.0)) / (192.0*E*material.I)
    mid_span_moment = (LOAD*L) / 8.0
    support_moment = -(LOAD*L) / 8.0
    for n in range(0, nodal_coordinates.shape[0]):
        x_coord = nodal_coordinates[n, 0]
        if ((x_coord <= SPATIAL_TOLERANCE) or (abs(x_coord - L) <= SPATIAL_TOLERANCE)):
            assert abs(system.internal_forces[n, 2] - support_moment) < NUMERICAL_TOLERANCE, \
                f"Error in bending moment: expected {support_moment} but computed {system.internal_forces[n, 2]}."
        elif (abs(x_coord - (0.50 * L)) <= SPATIAL_TOLERANCE):
            assert abs(system.state[n, 1] - analytical_displacement) < NUMERICAL_TOLERANCE, \
                f"Error in X displacement: expected {analytical_displacement} but computed {system.state[n, 1]}."
            assert abs(system.internal_forces[n, 2] - mid_span_moment) < NUMERICAL_TOLERANCE, \
                f"Error in bending moment: expected {mid_span_moment} but computed {system.internal_forces[n, 2]}."

if __name__ == "__main__":

    # run the tests
    test_simply_supported()
    test_cantilever()
    test_bar_tension()
    test_fixed_fixed()