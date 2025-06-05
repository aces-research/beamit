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
# critical effective cohesive strength
Sc = 400.0E06
# number of elements
Nel = 100

# applied loads and tolerances
SPALL_VELOCITY = (0.50*Sc)/(rho*np.sqrt(E/rho))
SPATIAL_TOLERANCE = 1.0E-05

# time step details
dt = 1.0E-09
time_steps = 10000
save_time = 10

# Function to update BCs
def update_BCs(nodal_coordinates, simulation_time, bcvalues):
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # axial displacement at left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bcvalues[i, 0] = -SPALL_VELOCITY*simulation_time
        # axial displacement at right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bcvalues[i, 0] = SPALL_VELOCITY*simulation_time

if __name__ == "__main__":

    start_time = time.time()

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

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
    initial_velocity = np.zeros([function_space.N, function_space.dof])
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        # roller and axial displacement at left end
        if (x_coord <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            initial_velocity[i, 0] = -SPALL_VELOCITY
        # roller and axial displacement at right end
        elif (abs(x_coord - L) <= SPATIAL_TOLERANCE):
            bctypes[i, 0] = 1
            bctypes[i, 1] = 1
            initial_velocity[i, 0] = SPALL_VELOCITY
    
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

    # solve the dynamic problem and update the state of system
    simulation_time = 0.0
    for i in range(0, time_steps):
        print("\nCurrent time step:", i+1,"out of", time_steps, "time steps.")
        simulation_time += dt
        update_BCs(nodal_coordinates, simulation_time, bcvalues)
        # apply the boundary conditions
        solver.modify_boundary_condition_values(bcvalues)
        # solve the dynamic problem
        solver.solve(dt)
        if ((i+1) % save_time == 0):
            output_file = "./VTK/output-" + str(i+1)
            PostProcess.write_output_vtk(output_file, system)

    print("\nTotal simulation time = %.3f sec." % (time.time() - start_time))