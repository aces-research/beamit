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
R = 0.1
# length of beam
L = 20.0
# number of elements
Nel = 10
# applied loads and tolerances
MOMENT = 2.50E+06
NUMBER_LOAD_STEPS = 10
SPATIAL_TOLERANCE = 1.0E-05
NUMERICAL_TOLERANCE = 1.0E-02

def test_beamDG_circle():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.TFKLGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type="DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics)
    system = System.System(function_space, material)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at the left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]

    solver.set_boundary_conditions(bctypes, bcvalues)

    for load_step in range(0, NUMBER_LOAD_STEPS):
        print("\nCurrent load step =", load_step + 1,
              "out of", NUMBER_LOAD_STEPS, "load steps.")
        load_level = (load_step + 1) / NUMBER_LOAD_STEPS
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]):
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # bending moment at the right end
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 5] = load_level * MOMENT
        solver.modify_boundary_condition_values(bcvalues)
        # solve the problem
        solver.solve(Nmax=10, tol=1.0E-08)

    # test the bending moments
    for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
        assert abs(system.internal_forces[n, 5] - MOMENT) / MOMENT < NUMERICAL_TOLERANCE, \
            f"Error in bending moment: expected {MOMENT} but computed {system.internal_forces[n, 5]}."

if __name__ == "__main__":
    test_beamDG_circle()