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
L = 0.1
# number of elements
Nel = 10
# critical effective cohesive strength
Sc = 400.0E06
# effective fracture energy
Gc = 34.0

# applied loads and tolerances
BENDING_DISPLACEMENT = -0.005
SPATIAL_TOLERANCE = 1.0E-05

# the load steps and output
load_steps = 100
save_step = 1

# function to update the boundary conditions
def update_BCs(bcvalues, initial_state, load_level):
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # bending displacement at the center
        if ((abs(x_coord - (L/2.0)) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 1] = initial_state[i, 1] + load_level * BENDING_DISPLACEMENT

if __name__ == "__main__":

    start_time = time.time()

    # physical information (material parameters)
    material = Material.CohesiveInterfaceMaterial(rho, E, R, Sc, Gc)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics)
    system = System.System(function_space, material)
    initial_state = copy.deepcopy(system.state)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # set the boundary conditions
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at the left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]
        # clamp at the right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
            bctypes[i, 4:6] = 1
            bcvalues[i, 4:6] = initial_state[i, 4:6]
        # Dirichlet dof at the center
        elif ((abs(x_coord - (L/2.0)) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 1] = 1
            bcvalues[i, 1] = initial_state[i, 1]

    solver.set_boundary_conditions(bctypes, bcvalues)

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
        print("\nCurrent load step:", load_step+1,"out of", load_steps, "load steps.")
        load_level = (load_step+1)/load_steps
        # update the boundary conditions
        update_BCs(bcvalues, initial_state, load_level)
        # solve the quasi-static problem, update the system state and write the result
        solver.modify_boundary_condition_values(bcvalues)
        solver.solve(Nmax = 10, tol = np.abs(BENDING_DISPLACEMENT*1.0E-03))
        if ((load_step+1) % save_step == 0):
            output_file = "./VTK/output-" + str(load_step+1)
            PostProcess.write_output_vtk(output_file, system)

    print("\nTotal simulation time = %.2f sec." % (time.time() - start_time))