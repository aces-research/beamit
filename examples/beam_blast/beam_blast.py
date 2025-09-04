from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy

# density of the material
rho = 2693.0
# elastic modulus of beam
E = 68.9E9
# Poisson's ratio of beam
nu = 0.343
# length of beam
L = 0.2032
# width of the beam
W = 0.0254
# height of the beam
H = 0.00635
# number of elements
Nel = 100
# applied load
IMPULSE = 1.78E03
VELOCITY = IMPULSE / (rho * H)
# spatial tolerance
SPATIAL_TOLERANCE = 1.0E-10

# the time steps and output
dt = 1.0E-08
final_time = 2.0E-03
vtk_dump = 1000

if __name__ == "__main__":

    # physical information (material parameters)
    material = Material.ShearFlexibleMaterial(
        rho, E, nu, A=W*H, I=(W*H**3)/12, I_minor=(H*W**3)/12)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type="DG")
    function_space.discretize()
    # to avoid creating reference to the object attributes
    # a better idea is to create private attributes and use accessors
    nodal_coordinates = copy.deepcopy(function_space.nodes)

    # a system binding the function_space (math) and the material (physics)
    system = System.System(function_space, material)
    # to avoid creating reference to the object attributes
    initial_state = copy.deepcopy(system.state)

    # solver
    solver = Solver.ExplicitNewmarkSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])
    # initial velocity matrix
    initial_velocity = np.zeros([function_space.N, function_space.dof])

    # generate and set boundary and initial conditions
    for i in range(0, nodal_coordinates.shape[0]):  # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp at left end
        if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:6] = 1
            bcvalues[i, 0:6] = initial_state[i, 0:6]
        # clamp at right end
        elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
            bctypes[i, 0:6] = 1
            bcvalues[i, 0:6] = initial_state[i, 0:6]
        # initial velocity for rest of the nodes
        else:
            initial_velocity[i, 2] = -VELOCITY

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

    # solve the problem
    current_time = 0.0
    time_steps = 0
    while (current_time < final_time):
        time_steps += 1
        current_time += dt
        print("\nCurrent time =", '{:0.5e}'.format(current_time), "out of", final_time, "sec.")
        # solve for one time step
        solver.solve(dt)
        # write the output
        if (time_steps % vtk_dump == 0):
            output_file = "./VTK/output-" + str(time_steps)
            PostProcess.write_output_vtk(output_file, system)
