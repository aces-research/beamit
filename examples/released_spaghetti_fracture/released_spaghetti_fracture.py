from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy
import time

# density of the material
rho = 1400.0
# elastic modulus of beam
E = 5.50E09
# the radius of the beam
R = 5.70E-04
# length of beam
L = 0.24
# number of elements
Nel = 10

# applied loads and tolerances
INITIAL_MOMENT = 0.006466
SPATIAL_TOLERANCE = 1.0E-08

# the load / time steps and output
load_steps = 10
static_dump = 10
dt = 1.0E-06
final_time = 1.0E-02
dynamic_dump = 100

if __name__ == "__main__":

    start_time = time.time()

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.TFKLGeometricallyExactFunctionSpace(0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    ############### THE QUASI-STATIC STAGE ###############

    print("\n############### THE QUASI-STATIC STAGE ###############")

    # the static solver
    static_solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]

    static_solver.set_boundary_conditions(bctypes, bcvalues)

    # create a VTK directory or clear it
    if not os.path.isdir("VTK"):
        os.mkdir("VTK")
    else:
        for item in os.listdir("VTK"):
            os.remove(os.path.join("VTK", item))

    # write the initial results
    PostProcess.write_output_vtk("./VTK/output-0", system)

    # solve the static problem
    for i in range(0, load_steps):
        print("\nCurrent load step:", i+1,"out of", load_steps, "load steps.")
        load_level = (i+1)/load_steps
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]): # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # moment at the right end
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 5] = load_level * INITIAL_MOMENT
        static_solver.modify_boundary_condition_values(bcvalues)
        # solve the nonlinear static problem and update the system
        static_solver.solve(Nmax=10, tol=1.0E-10)
        if ((i+1) % static_dump == 0):
            output_file = "./VTK/output-" + str(i+1)
            PostProcess.write_output_vtk(output_file, system)
    
    ############### THE DYNAMIC STAGE ###############

    print("\n############### THE DYNAMIC STAGE ###############")

    # get the deformed state of the beam
    deformed_state = copy.deepcopy(system.state)

    # the dynamic solver
    dynamic_solver = Solver.ExplicitNewmarkSolver(system)

    # recreate and set the boundary conditions
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    bcvalues = np.zeros([function_space.N, function_space.dof])
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]

    dynamic_solver.set_boundary_conditions(bctypes, bcvalues)

    # generate initial velocities and set initial conditions
    initial_velocity = np.zeros([function_space.N, function_space.dof])
    dynamic_solver.set_initial_conditions(deformed_state, initial_velocity)

    current_time = 0.0
    time_steps = 0

    while (current_time < final_time):
        time_steps += 1
        current_time += dt
        print("\nCurrent time =", '{:0.5e}'.format(current_time), "out of", final_time, "sec.")
        # solve the dynamic problem
        dynamic_solver.solve(dt)
        # write the output
        if (time_steps % dynamic_dump == 0):
            output_file = "./VTK/output-" + str(load_steps + time_steps)
            PostProcess.write_output_vtk(output_file, system)

    print("\nTotal simulation time = %.2f sec." % (time.time() - start_time))
