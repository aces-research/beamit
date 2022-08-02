from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os

def static_main():

    # density of the material
    rho = 7850.0
    # elastic modulus of beam
    E = 2.0e11
    # area of cross section
    A = 0.0314159
    # area moment of inertia
    I = 7.85398e-5
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

    # length of beam
    L = 1
    # number of elements
    Nel = 5
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "CG")
    function_space.discretize()
    nodal_coordinates = function_space.nodes

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewtonRaphsonSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # clamp left node (a clamp fixes both the position and the direction of the tangent)
    # Note that the tangent vector r' is a unit vector if and only if the axial strain is zero.
    # Therefore imposing that the tangent vector is [1, 0, 0] implies that both 
    # 1) the tangent is horizontal, and 
    # 2) the strain at the clamped end is zero
    # However, in general we want to only impose condition 1) as we do not know the strain at the 
    # clamped end a priori. Therefore imposing a horizontal tangent (e.g. aligned with the x-axis) 
    # is achieved by imposing that the y and z component of the tangent be fixed and equal to 0,
    # whereas the x component of the tangent is free (zero Neumann boundary condition). 
    # See also Meier 2014, CMAME for more details.  
    for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
        x_coord = nodal_coordinates[i, 0]
        y_coord = nodal_coordinates[i, 1]
        z_coord = nodal_coordinates[i, 2]
        # clamp left node (a clamp fixes both the position and the direction of the tangent)
        if ((x_coord == 0.0) and (y_coord == 0.0) and (z_coord == 0.0)):
            bctypes[i, 0:3] = 1
            bctypes[i, 4:6] = 1
            bcvalues[i, 0:3] = system.state[i, 0:3]
            bcvalues[i, 4:6] = system.state[i, 4:6]
    # apply constant force on right most node  
    bcvalues[Nel, 2] = -1.0e04

    # apply the boundary conditions
    solver.set_boundary_conditions(bctypes, bcvalues)

    # create a VTK directory or clear it
    if not os.path.isdir("VTK"):
        os.mkdir("VTK")
    else:
        for item in os.listdir("VTK"):
            os.remove(os.path.join("VTK", item))

    output_file = "./VTK/output"

    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve(100, 1.0E-05)

    PostProcess.write_positions_vtk(output_file, system)

def dynamic_main():

    # density of the material
    rho = 7850.0
    # elastic modulus of beam
    E = 2.0e11
    # area of cross section
    A = 0.0314159
    # area moment of inertia
    I = 7.85398e-5
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

    # length of beam
    L = 1
    # number of elements
    Nel = 5
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "CG")
    function_space.discretize()
    nodal_coordinates = function_space.nodes

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # the solver
    solver = Solver.NewmarkSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # function to apply BCs
    def get_supports(bctypes, bcvalues, load_case):
        if ((load_case == 0) or (load_case == 1)):
            for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
                x_coord = nodal_coordinates[i, 0]
                y_coord = nodal_coordinates[i, 1]
                z_coord = nodal_coordinates[i, 2]
                # clamp left node (a clamp fixes both the position and the direction of the tangent)
                if ((x_coord == 0.0) and (y_coord == 0.0) and (z_coord == 0.0)):
                    bctypes[i, 0:3] = 1
                    bctypes[i, 4:6] = 1
                    bcvalues[i, 0:3] = system.state[i, 0:3]
                    bcvalues[i, 4:6] = system.state[i, 4:6]

    # applied loads
    force_x = 0.0
    force_y = 0.0
    force_z = -1.0e4
    moment_y = 1.0e7

    # the load case
    load_case = 1

    # create a VTK directory or clear it
    if not os.path.isdir("VTK"):
        os.mkdir("VTK")
    else:
        for item in os.listdir("VTK"):
            os.remove(os.path.join("VTK", item))
    
    # write the initial positions
    output_file = "./VTK/output-0"
    PostProcess.write_positions_vtk(output_file, system)
    
    # solve the nonlinear dynamic problem and update the nodal position in system 
    time_steps = 10000 # No. of time steps
    save_time = 100
    get_supports(bctypes, bcvalues, load_case)
    for i in range(0, time_steps):
        print("\nCurrent time step:", i+1,"out of", time_steps, "time steps.")
        output_file = "./VTK/output-" + str(i+1)
        # apply loads
        if (load_case == 0):
            # apply constant force on right most node
            bcvalues[Nel, 0] = force_x*((i+1)/time_steps)
            bcvalues[Nel, 1] = force_y*((i+1)/time_steps)
            bcvalues[Nel, 2] = force_z*((i+1)/time_steps)
        elif (load_case == 1):
            # apply constant moment on right most node
            bcvalues[Nel, 4] = moment_y*((i+1)/time_steps)
        # apply the boundary conditions
        solver.set_boundary_conditions(bctypes, bcvalues)
        solver.solve(dt = 0.01, Nmax = 50)
        if ((i+1) % save_time == 0):
            PostProcess.write_positions_vtk(output_file, system)

# run main functions
# static_main()
dynamic_main()