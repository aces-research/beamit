from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy

# density of the material
rho = 1000.0
# elastic modulus of beam
E = 1.0E07
# Poisson's ratio of beam
nu = 0.0
# length of beam (45 degree bend of radius 100)
L = (100.0*np.pi)/4
# side of the square cross-section
a = 1.0
# moment of inertia
I = (a**4) / 12.0
# number of elements
Nel = 10
# applied loads and tolerances
INITIAL_TIP_MOMENT = (-1.0 * E * I * np.pi) / (4.0 * L)
TIP_LOAD = 2000.0
SPATIAL_TOLERANCE = 1.0E-10

# the load / time steps and output
load_steps = 100
vtk_dump = 1

def run_arc_segment_simulation(discretization_type):
    print("\nRunning the simulation with", discretization_type, "discretization")

    # physical information (material parameters)
    material = Material.ShearFlexibleMaterial(
        rho, E, nu, A=a**2, I=(a**4)/12, I_minor=(a**4)/12)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type=discretization_type)
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics)
    system = System.System(function_space, material, betaP=10.0*E, betaT=10.0*E)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    # solver
    solver = Solver.NewtonRaphsonSolver(system)

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

    solver.set_boundary_conditions(bctypes, bcvalues)

    # output directory
    output_dir = f"./VTK"

    # create the output directory or clear it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        for item in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, item))

    # write the initial results
    PostProcess.write_output_vtk(f"{output_dir}/output-0", system)

    # apply the preload
    print("\nApplying the preload...")
    for i in range(0, 10):
        print("\nCurrent pre-load step:", i+1, "out of", 10, "pre-load steps.")
        load_level = (i+1)/10
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # apply the preload at the tip
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 5] = load_level * INITIAL_TIP_MOMENT
        solver.modify_boundary_condition_values(bcvalues)
        # solve the problem and update the system
        solver.solve(Nmax=100, tol=1.0E-08)
    output_file = f"{output_dir}/output-1"
    PostProcess.write_output_vtk(output_file, system)

    # apply the tip load
    print("\nApplying the tip load...")
    for i in range(0, load_steps):
        print("\nCurrent load step:", i+1, "out of", load_steps, "load steps.")
        load_level = (i+1)/load_steps
        # update the boundary conditions
        for n in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
            x_coord = nodal_coordinates[n, 0]
            y_coord = nodal_coordinates[n, 1]
            z_coord = nodal_coordinates[n, 2]
            # apply the preload at the tip
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 2] = load_level * TIP_LOAD
        solver.modify_boundary_condition_values(bcvalues)
        # solve the problem and update the system
        solver.solve(Nmax=100, tol=1.0E-07)
        if ((i+1) % vtk_dump == 0):
            output_file = f"{output_dir}/output-" + str(i+2)
            PostProcess.write_output_vtk(output_file, system)

if __name__ == "__main__":
    # run the simulations
    run_arc_segment_simulation("CG")
