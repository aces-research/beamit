from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy

# density of the material
rho = 1100.0
# elastic modulus of beam
E = 1350.0E06
# poisson's ratio of beam
nu = 0.0
# length of beam
L = 1.0
# radius of beam
R = 0.006
# cross-section area of beam
A = np.pi * R**2.0
# moment of inertia of beam
I = (np.pi/4.0) * R**4.0
# cohesive strength
Sc = 40.0E06
# fracture energy
Gc = 10.0
# number of elements
Nel = 100

# applied moment
MOMENT = (2.0 * np.pi**2.0 * E * R**4.0) / L
# loading rate
LOADING_RATE = 1.0E-03
# spatial tolerance
SPATIAL_TOLERANCE = 1.0E-10

# the load / time steps and output
load_steps = 100
dt = 1.0E-06
final_time = 100.0
statics_vtk_dump = 10
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

    ############# STAGE 1: BEND THE BEAM INTO A DOUBLE CIRCLE #############

    # statics solver
    statics_solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:6] = 1
            bcvalues[i, 0:6] = initial_state[i, 0:6]

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
    for i in range(0, load_steps):
        print("\nCurrent load step:", i+1, "out of", load_steps, "load steps.")
        load_level = (i+1)/load_steps
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # apply bending moment at the right end
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 5] = load_level * MOMENT
        statics_solver.modify_boundary_condition_values(bcvalues)
        # solve and update the system
        statics_solver.solve(Nmax=100, tol=1.0E-06)
        if ((i+1) % statics_vtk_dump == 0):
            output_file = f"{output_dir}/output-" + str(i+1)
            PostProcess.write_output_vtk(output_file, system)

    ############# STAGE 2: APPLY Z-DISPLACEMENT AT THE RIGHT END #############

    # dynamics solver
    dynamics_solver = Solver.ExplicitNewmarkSolver(system)

    # initial velocity matrix
    initial_velocity = np.zeros([function_space.N, function_space.dof])

    # set the boundary and initial conditions
    post_bending_state = copy.deepcopy(system.state)
    for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
        x_coord = nodal_coordinates[n, 0]
        y_coord = nodal_coordinates[n, 1]
        z_coord = nodal_coordinates[n, 2]
        # Dirichlet BC at right end to apply z-displacement
        if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[n, 2] = 1
            bcvalues[n, 2] = post_bending_state[n, 2]

    dynamics_solver.set_boundary_conditions(bctypes, bcvalues)
    dynamics_solver.set_initial_conditions(post_bending_state, initial_velocity)

    # solve the problem
    current_time = 0.0
    time_steps = 0
    while (current_time < final_time):
        time_steps += 1
        current_time += dt
        print("\nCurrent time =", '{:0.5e}'.format(current_time), "out of", final_time, "sec.")
        # update the displacement at the right end
        for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # apply z-displacement at the right end
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 2] += LOADING_RATE * dt
        dynamics_solver.modify_boundary_condition_values(bcvalues)
        # solve for one time step
        dynamics_solver.solve(dt)
        # write the output
        if (time_steps % dynamics_vtk_dump == 0):
            output_file = f"{output_dir}/output-" + str(load_steps + time_steps)
            PostProcess.write_output_vtk(output_file, system)
