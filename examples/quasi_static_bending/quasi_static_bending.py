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
R = 7.0E-04
# critical effective cohesive strength
Sc = 19.0E06
# effective fracture energy
Gc = 100.0
# physical information (material parameters)
material = Material.CohesiveInterfaceMaterial(rho, E, R, Sc, Gc)

# length of beam
L = 24.0E-02
# number of elements
Nel = 50
# geometric information (domain, no. of elements)
function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "DG")
function_space.discretize()
# to avoid creating reference to the object attributes
nodal_coordinates = copy.deepcopy(function_space.nodes)

# a system binding the function_space (math) and the material (physics)
system = System.System(function_space, material)
initial_state = copy.deepcopy(system.state)

# the solvers
solver = Solver.NewtonRaphsonSolver(system)

# boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
# boundary condition values matrix
bcvalues = np.zeros([function_space.N, function_space.dof])

# applied loads and tolerances
SPATIAL_TOLERANCE = 1.0E-05
BENDING_DISPLACEMENT = 1.0E-02

# the load steps
load_steps = 200

# create a VTK directory or clear it
if not os.path.isdir("VTK"):
    os.mkdir("VTK")
else:
    for item in os.listdir("VTK"):
        os.remove(os.path.join("VTK", item))

# write the initial results
output_file = "./VTK/output-0"
PostProcess.write_output_vtk(output_file, system)

for load_step in range(0, load_steps):
    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # pin at the left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]
        # roller at the right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]
        # bending load at the center
        elif ((abs(x_coord - (L/2.0)) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 1] = 1
            bcvalues[i, 1] += -(BENDING_DISPLACEMENT/load_steps)

    # solve the quasi-static problem, update the system state and write the result
    solver.set_boundary_conditions(bctypes, bcvalues)
    solver.solve(Nmax = 10, tol = BENDING_DISPLACEMENT*1.0E-03)
    PostProcess.write_output_vtk("./VTK/output-"+str(load_step+1), system)

print("\nTotal simulation time = %.2f sec." % (time.time() - start_time))