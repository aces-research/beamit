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
    E = 2.0E11
    # area of cross section
    A = 0.0314159
    # area moment of inertia
    I = 7.85398E-05
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
    E = 2.0E11
    # area of cross section
    A = 0.0314159
    # area moment of inertia
    I = 7.85398E-05
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

    # length of beam
    L = 1
    # number of elements
    Nel = 2
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "CG")
    function_space.discretize()
    nodal_coordinates = function_space.nodes

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)
    initial_state = system.state

    # the solver
    solver = Solver.NewmarkSolver(system)

    # boundary condition types (0 = Neumann, 1 = Dirichlet) matrix
    bctypes = np.zeros([function_space.N, function_space.dof], dtype=np.int64)
    # boundary condition values matrix
    bcvalues = np.zeros([function_space.N, function_space.dof])

    # applied loads and tolerances
    FORCE_X_CB = -5.0E05
    FORCE_Y = -1.0E04
    FORCE_Z = -1.0E04
    MOMENT_Z = 1.0E03
    PERTURB_CB = 100.0
    SPATIAL_TOLERANCE = 1.0E-05
      
    # function to apply BCs
    def update_BCs(bctypes, bcvalues, load_case, load_level):
        for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
            x_coord = nodal_coordinates[i, 0]
            y_coord = nodal_coordinates[i, 1]
            z_coord = nodal_coordinates[i, 2]
            if (load_case == 0):
                # clamp left node
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bctypes[i, 4:6] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                    bcvalues[i, 4:6] = initial_state[i, 4:6]
                # apply force on right node
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 2] = load_level*FORCE_Z
            elif (load_case == 1):
                # clamp left node
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bctypes[i, 4:6] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                    bcvalues[i, 4:6] = initial_state[i, 4:6]
                # apply moment on right node
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 5] = load_level*MOMENT_Z
            elif(load_case == 2):
                # pin at left end
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                # roller at the right end
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 1:3] = 1
                    bcvalues[i, 1:3] = initial_state[i, 1:3]
                # apply force at the center
                elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 1] = load_level*FORCE_Y
            elif(load_case == 3):
                # pin at left end
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                # roller and axial load at the right end
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 1:3] = 1
                    bcvalues[i, 1:3] = initial_state[i, 1:3]
                    bcvalues[i, 0] = load_level*FORCE_X_CB
                # perturbation force at the center
                elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 1] = PERTURB_CB

    # the load case
    load_case = 3

    # set boundary and initial conditions
    update_BCs(bctypes, bcvalues, load_case, load_level = 0.0)
    solver.set_boundary_conditions(bctypes, bcvalues)
    initial_position = initial_state
    initial_velocity = np.zeros([function_space.N, function_space.dof])
    solver.set_initial_conditions(initial_position, initial_velocity)
    
    # create a VTK directory or clear it
    if not os.path.isdir("VTK"):
        os.mkdir("VTK")
    else:
        for item in os.listdir("VTK"):
            os.remove(os.path.join("VTK", item))
    
    # write the initial displacements
    output_file = "./VTK/output-0"
    PostProcess.write_displacements_vtk(output_file, system)
    
    # solve the nonlinear dynamic problem and update the nodal position in system 
    time_steps = 10000 # No. of time steps
    save_time = 100
    for i in range(0, time_steps):
        print("\nCurrent time step:", i+1,"out of", time_steps, "time steps.")
        output_file = "./VTK/output-" + str(i+1)
        load_level = (i+1)/time_steps
        update_BCs(bctypes, bcvalues, load_case, load_level)
        # apply the boundary conditions
        solver.set_boundary_conditions(bctypes, bcvalues)
        solver.solve(dt = 0.01, Nmax = 20, tol = 1.0E-03)
        if ((i+1) % save_time == 0):
            PostProcess.write_displacements_vtk(output_file, system)

# run main functions
# static_main()
dynamic_main()