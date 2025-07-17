from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy

# density of the material
rho = 7850.0
# elastic modulus of beam
E = 2.0E11
# Poisson's ratio of beam
nu = 0.30
# length of beam
L = 1.0
# beam slenderness ratio
slenderness_ratio = 10.0
# side of the square cross-section
a = L / slenderness_ratio
# number of elements
NEls = [8, 16, 32, 64, 128]
# applied loads and tolerances
TIP_MOMENT = 1.0E07
SPATIAL_TOLERANCE = 1.0E-10

# the load / time steps and output
load_steps = 1000
vtk_dump = 100

def run_pure_bending_simulation(discretization_type, Nel):
    print("\nRunning the simulation with", discretization_type, 
          "discretization and", Nel, "elements.")

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
    system = System.System(function_space, material)
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
    output_dir = f"./VTK-{discretization_type}/{Nel}els"

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
            # apply a moment couple at the right end
            if ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                bcvalues[n, 3] = load_level * TIP_MOMENT
                bcvalues[n, 5] = load_level * TIP_MOMENT
        solver.modify_boundary_condition_values(bcvalues)
        # solve the problem and update the system
        solver.solve(Nmax=100, tol=1.0E-06)
        if ((i+1) % vtk_dump == 0):
            output_file = f"{output_dir}/output-" + str(i+1)
            PostProcess.write_output_vtk(output_file, system)

if __name__ == "__main__":

    for i in range(len(NEls)):
        # run the simulation for each number of elements
        run_pure_bending_simulation("CG", NEls[i])
