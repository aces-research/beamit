from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy

# density of the material
rho = 3690.0
# elastic modulus of beam
E = 260.0E09
# poisson's ratio of beam
nu = 0.30
# length of beam
L = 1.0
# side of the square cross-section
a = 0.01
# cross-section area of beam
A = a**2
# moment of inertia of beam
I = (a**4) / 12.0
# cohesive strength
Sc = 400.0E06
# fracture energy
Gc = 34.0
# number of elements
Nel = 100

# applied loads and tolerances
LOAD_STEP = 0.01
DYNAMICS_SWITCH_FACTOR = 0.95
LOADING_RATE = 1.0E-03
SPATIAL_TOLERANCE = 1.0E-10

# the load / time steps and output
dt = 1.0E-06
final_time = 100.0
statics_vtk_dump = 1
dynamics_vtk_dump = 10000

if __name__ == "__main__":

    # physical information (material parameters)
    material = Material.ShearFlexibleCohesiveInterfaceMaterial(
        rho, E, nu, A=A, I=I, I_minor=I, Sc=Sc, Gc=Gc)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type="DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics)
    system = System.System(function_space, material)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    ############# STAGE 1: APPLY DISPLACEMENT AND ROTATION STATICALLY #############

    # statics solver
    statics_solver = Solver.NewtonRaphsonSolver(system)

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:6] = 1
            bcvalues[i, 0:6] = initial_state[i, 0:6]
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 1] = 1
            bcvalues[i, 1] = initial_state[i, 1]
            bctypes[i, 2] = 1
            bcvalues[i, 2] = initial_state[i, 2]
            bctypes[i, 3] = 1
            bcvalues[i, 3] = initial_state[i, 3]

    statics_solver.set_boundary_conditions(bctypes, bcvalues)

    # output directory
    output_dir = "./VTK"

    # create the output directory or clear it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        for item in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, item))

    # write the initial results
    PostProcess.write_output_vtk(f"{output_dir}/output-0", system)

    # solve the problem
    load_step = 1
    while True:
        print(f"\nThis is load step #{load_step}.")
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # apply displacement and rotation at the right end
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 1] += LOAD_STEP
                bcvalues[n, 2] += LOAD_STEP
                bcvalues[n, 3] += LOAD_STEP
        statics_solver.modify_boundary_condition_values(bcvalues)
        # solve and update the system
        statics_solver.solve(Nmax=100, tol=1.0E-04)
        load_step += 1
        if ((load_step) % statics_vtk_dump == 0):
            output_file = f"{output_dir}/output-" + str(load_step)
            PostProcess.write_output_vtk(output_file, system)
        # check if we have almost reached the cohesive strength of material
        # from effective force at the interface
        if (np.any(system.weak_form.internal_variables[:, 3:4] >= DYNAMICS_SWITCH_FACTOR * Sc * A)):
            print("\nAlmost reached the cohesive strength of the material. Switching to dynamics...")
            break
