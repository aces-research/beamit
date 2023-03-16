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
rho = 1500.0
# elastic modulus of beam
E = 3.80E09
# the radius of the beam
R = 7.0E-06
# critical effective cohesive strength
Sc = 19.0E06
# effective fracture energy
Gc = 10.0
# physical information (material parameters)
material = Material.CohesiveInterfaceMaterial(rho, E, R, Sc, Gc)

# length of beam
L = 24.0E-04
# number of elements
Nel = 100
# geometric information (domain, no. of elements)
function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "DG")
function_space.discretize()
# to avoid creating reference to the object attributes
nodal_coordinates = copy.deepcopy(function_space.nodes)

# a system binding the function_space (math) and the material (physics)
system = System.System(function_space, material)
initial_state = copy.deepcopy(system.state)

# the solvers
quasi_static_solver = Solver.NewtonRaphsonSolver(system)
# dynamic_solver = Solver.ExplicitNewmarkSolver(system)
dynamic_solver = Solver.ImplicitNewmarkSolver(system)

# boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
# boundary condition values matrix
bcvalues = np.zeros([function_space.N, function_space.dof])

# applied loads and tolerances
SPATIAL_TOLERANCE = 1.0E-05
BENDING_LOAD = 0.62E-04
QUENCH_RATE = 5.0E-05

# the load steps
load_steps = 35

# create a VTK directory or clear it
if not os.path.isdir("VTK"):
    os.mkdir("VTK")
else:
    for item in os.listdir("VTK"):
        os.remove(os.path.join("VTK", item))

# write the initial results
output_file = "./VTK/output-0"
PostProcess.write_output_vtk(output_file, system)

############### THE QUASI-STATIC STAGE ###############

print("\n############### THE QUASI-STATIC STAGE ###############")

for load_step in range(0, load_steps):
    # set the boundary conditions for the quasi-static stage
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # pin at the left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
        # roller at the right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 1:3] = 1
            bcvalues[i, 1:3] = initial_state[i, 1:3]
        # bending load at the center
        elif ((abs(x_coord - (L/2.0)) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 1] += -0.50*(BENDING_LOAD/load_steps)

    # solve the quasi-static problem, update the system state and write the result
    quasi_static_solver.set_boundary_conditions(bctypes, bcvalues)
    quasi_static_solver.solve(Nmax = 10, tol = BENDING_LOAD*1.0E-03)
    PostProcess.write_output_vtk("./VTK/output-"+str(load_step+1), system)

############### THE DYNAMIC STAGE ###############

print("\n############### THE DYNAMIC STAGE ###############")

# get the deformed state of the beam
deformed_state = copy.deepcopy(system.state)

# update and set the boundary conditions for the dynamic stage
for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
    x_coord = nodal_coordinates[i, 0]
    y_coord = nodal_coordinates[i, 1]
    z_coord = nodal_coordinates[i, 2]
    # pin at left end
    if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        bctypes[i, 0:3] = 1
        bcvalues[i, 0:3] = deformed_state[i, 0:3]
    # pin at right end
    elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        bctypes[i, 0:3] = 1
        bcvalues[i, 0:3] = deformed_state[i, 0:3]

dynamic_solver.set_boundary_conditions(bctypes, bcvalues)

# generate initial positions and velocities
initial_velocity = np.zeros([function_space.N, function_space.dof])

# set the initial conditions for the dynamic stage
for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
    x_coord = nodal_coordinates[i, 0]
    y_coord = nodal_coordinates[i, 1]
    z_coord = nodal_coordinates[i, 2]
    # at left end
    if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        initial_velocity[i, 0] = QUENCH_RATE
    # at right end
    elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
        initial_velocity[i, 0] = -QUENCH_RATE

dynamic_solver.set_initial_conditions(deformed_state, initial_velocity)

# function to update dynamic BCs
def update_BCs(analysis_time):
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # axial displacement at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 0] = deformed_state[i, 0] + (QUENCH_RATE*analysis_time)
        # axial displacement at right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 0] = deformed_state[i, 0] - (QUENCH_RATE*analysis_time)

# the time details
dt = 1.0E-09
# dt = dynamic_solver.stable_time_step
time_steps = 2000
save_time = 1

# solve the dynamic problem and update the state of system
analysis_time = 0.0
for time_step in range(0, time_steps):
    print("\nCurrent time step:", time_step+1,"out of", time_steps, "time steps.")
    analysis_time += dt
    update_BCs(analysis_time)
    # apply the boundary conditions
    dynamic_solver.modify_boundary_condition_values(bcvalues)
    # dynamic_solver.solve(dt)
    dynamic_solver.solve(dt, tol=QUENCH_RATE*1.0E-03, Nmax=20)
    if ((time_step+1) % save_time == 0):
        output_file = "./VTK/output-" + str(load_steps+time_step+1)
        PostProcess.write_output_vtk(output_file, system)

print("\nTotal simulation time = %.2f sec." % (time.time() - start_time))