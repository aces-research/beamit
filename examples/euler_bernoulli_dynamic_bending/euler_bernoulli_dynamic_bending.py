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
rho = 3690.0
# elastic modulus of beam
E = 2.60E11
# the radius of the beam
R = 1.0E-03
# length of beam
L = 0.10
# number of elements
Nel = 10

# applied loads and tolerances
LOADING_RATE = 0.01
SPATIAL_TOLERANCE = 1.0E-05

# time step details
dt = 1.0E-07
rise_time = 1.0E-04
final_time = 2.0E-01
save_time = 1000

# Function to update BCs
def update_BCs(nodal_coordinates, simulation_time, bcvalues):
    if (simulation_time <= rise_time):
        applied_velocity = (LOADING_RATE*simulation_time)/rise_time
    else:
        applied_velocity = LOADING_RATE
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # transverse displacement at the center
        if (abs(x_coord - (0.50 * L)) <= SPATIAL_TOLERANCE):
            bcvalues[i, 1] = -applied_velocity*simulation_time

if __name__ == "__main__":

    start_time = time.time()

    # physical information (material parameters)
    material = Material.Material(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
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

    # generate and set boundary and initial conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # clamp at left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            bctypes[i, 2] = 1
        # clamp at right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            bctypes[i, 2] = 1
        # transverse displacement at the center
        elif (abs(x_coord - (0.50 * L)) <= SPATIAL_TOLERANCE):
            bctypes[i, 1] = 1

    initial_velocity = np.zeros([function_space.N, function_space.dof])    
    solver.set_boundary_conditions(bctypes, bcvalues)
    solver.set_initial_conditions(initial_state, initial_velocity)

    # create a VTK directory or clear it
    if not os.path.isdir("VTK"):
        os.mkdir("VTK")
    else:
        for item in os.listdir("VTK"):
            os.remove(os.path.join("VTK", item))
        
    # write the initial results
    output_file = "./VTK/output-0"
    PostProcess.write_output_vtk(output_file, system)

    current_time = 0.0
    time_steps = 0

    while (current_time < final_time):
        time_steps += 1
        current_time += dt
        print("\nCurrent time =", '{:0.5e}'.format(current_time), "out of", final_time, "sec.")
        update_BCs(nodal_coordinates, current_time, bcvalues)
        # apply the boundary conditions
        solver.modify_boundary_condition_values(bcvalues)
        # solve the dynamic problem
        solver.solve(dt)
        # write the output
        if (time_steps % save_time == 0):
            output_file = "./VTK/output-" + str(time_steps)
            PostProcess.write_output_vtk(output_file, system)
    
    print("\nTotal simulation time = %.2f sec." % (time.time() - start_time))