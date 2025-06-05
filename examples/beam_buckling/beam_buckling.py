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
rho = 7850.0
# elastic modulus of beam
E = 2.0E11
# radius of the  beam
R = 0.1
# length of beam
L = 10.0
# number of elements
Nel = 10

# applied loads and tolerances
DISP_CB = -0.005
PERTURB_FORCE = 1.0
SPATIAL_TOLERANCE = 1.0E-05

# the load steps and output
load_steps = 1000
save_step = 1

# function to update the boundary conditions
def update_BCs(bcvalues, initial_state, load_level):
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # axial displacement at the right end
        if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bcvalues[i, 0] = initial_state[i, 0] + load_level * DISP_CB

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
        # pin at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
        # pin at the right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:3] = 1
            bcvalues[i, 0:3] = initial_state[i, 0:3]
        # perturbation force at the center
        elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            # apply equal forces on interface nodes
            bcvalues[i, 1] = 0.50*PERTURB_FORCE

    solver.set_boundary_conditions(bctypes, bcvalues)

    # create a VTK directory or clear it
    if not os.path.isdir("VTK"):
        os.mkdir("VTK")
    else:
        for item in os.listdir("VTK"):
            os.remove(os.path.join("VTK", item))

    # write the initial displacements
    PostProcess.write_output_vtk("./VTK/output-0", system)

    # solve the problem
    for i in range(0, load_steps):
        print("\nCurrent load step:", i+1,"out of", load_steps, "load steps.")
        load_level = (i+1)/load_steps
        # update the boundary conditions
        update_BCs(bcvalues, initial_state, load_level)
        solver.modify_boundary_condition_values(bcvalues)
        # solve the nonlinear static problem and update the system
        solver.solve(Nmax=10, tol=1.0E-03)
        if ((i+1) % save_step == 0):
            output_file = "./VTK/output-" + str(i+1)
            PostProcess.write_output_vtk(output_file, system)

    simulation_time = time.time() - start_time

    print("\nTotal simulation time = %.2f sec." % (simulation_time))