from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy
import time

start_time = time.time()

# density of the material
rho = 3690.0
# elastic modulus of beam
E = 2.60E11
# the radius of the beam
R = 5.0E-05
# critical effective cohesive strength
Sc = 400.0E06
# effective fracture energy
Gc = 34.0
# physical information (material parameters)
material = Material.TFKLCohesiveInterfaceMaterial(rho, E, R, Sc, Gc)

# length of beam
L = 0.005
# number of elements
Nel = 80
# geometric information (domain, no. of elements)
function_space = FunctionSpace.TFKLGeometricallyExactFunctionSpace(0, L, Nel, discretization_type = "DG")
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

# applied loads and tolerances
SPATIAL_TOLERANCE = 1.0E-05
SPALL_VELOCITY = (0.50*Sc)/(rho*np.sqrt(E/rho))

# generate and set boundary conditions
for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
    x_coord = nodal_coordinates[i, 0]
    y_coord = nodal_coordinates[i, 1]
    z_coord = nodal_coordinates[i, 2]
    # roller and axial displacement at left end
    if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        bctypes[i, 0:3] = 1
        bcvalues[i, 0:3] = initial_state[i, 0:3]
    # roller and axial displacement at right end
    elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        bctypes[i, 0:3] = 1
        bcvalues[i, 0:3] = initial_state[i, 0:3]

solver.set_boundary_conditions(bctypes, bcvalues)
initial_velocity = np.zeros([function_space.N, function_space.dof])

# generate initial velocities and set initial conditions
for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
    x_coord = nodal_coordinates[i, 0]
    y_coord = nodal_coordinates[i, 1]
    z_coord = nodal_coordinates[i, 2]
    # at left end
    if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        initial_velocity[i, 0] = -SPALL_VELOCITY
    # at right end
    elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        initial_velocity[i, 0] = SPALL_VELOCITY

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

# function to update BCs
def update_BCs(simulation_time):
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # roller and axial displacement at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 0] = initial_state[i, 0] - (SPALL_VELOCITY*simulation_time)
        # roller and axial displacement at right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 0] = initial_state[i, 0] + (SPALL_VELOCITY*simulation_time)

# the time details
dt = solver.stable_time_step*0.50
time_steps = 5000
save_time = 1

# solve the dynamic problem and update the state of system
simulation_time = 0.0
for i in range(0, time_steps):
    print("\nCurrent time step:", i+1,"out of", time_steps, "time steps.")
    simulation_time += dt
    update_BCs(simulation_time)
    # apply the boundary conditions
    solver.modify_boundary_condition_values(bcvalues)
    solver.solve(dt)
    if ((i+1) % save_time == 0):
        output_file = "./VTK/output-" + str(i+1)
        PostProcess.write_output_vtk(output_file, system)

analysis_time = time.time() - start_time

print("\nTotal simulation time = %.2f sec." % (analysis_time))