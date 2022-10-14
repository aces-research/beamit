from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
from beamit import PostProcess
import numpy as np
import os
import copy

def static_main():

    # density of the material
    rho = 7850.0
    # elastic modulus of beam
    E = 2.0E11
    # area of cross section
    A = 3.14159E-02
    # area moment of inertia
    I = 7.85398E-05
    # physical information (material parameters)
    material = Material.Material(rho, E, A, I)

    # length of beam
    L = 10.0
    # number of elements
    Nel = 2
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel, discretization_type = "DG")
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

    # the load case and output
    load_case = 0
    load_steps = 1000
    save_step = 1

    # applied loads and tolerances
    FORCE_Y = -1.0E04
    MOMENT_Z = 1.0E04
    PERTURB_FORCE = 100.0
    DISP_CB = -0.01
    SPATIAL_TOLERANCE = 1.0E-05

    # Function to generate the boundary conditions
    def get_BCs(bctypes, bcvalues, load_case, load_level = 0.0):
        for i in range(0, nodal_coordinates.shape[0]): # loop over the nodes
            x_coord = nodal_coordinates[i, 0]
            y_coord = nodal_coordinates[i, 1]
            z_coord = nodal_coordinates[i, 2]
            if (load_case == 0): # SIMPLY SUPPORTED BEAM WITH POINT LOAD AT THE CENTER
               # pin at left end
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                # roller at the right end
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 1:3] = 1
                    bcvalues[i, 1:3] = initial_state[i, 1:3]
                # apply load at the center
                elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 1] = FORCE_Y
            elif (load_case == 1): # CANTILEVER BEAM WITH POINT MOMENT AT THE RIGHT END
                # clamp left node
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bctypes[i, 4:6] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                    bcvalues[i, 4:6] = initial_state[i, 4:6]
                # apply moment on right node
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 5] = MOMENT_Z
            elif (load_case == 2): # FIXED-FIXED BEAM WITH POINT LOAD AT THE CENTER
                # clamp at left end
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bctypes[i, 4:6] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                    bcvalues[i, 4:6] = initial_state[i, 4:6]
                # clamp at right end
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bctypes[i, 4:6] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                    bcvalues[i, 4:6] = initial_state[i, 4:6]
                # apply force at the center
                elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 1] = FORCE_Y
            elif (load_case == 3): # SIMPLY SUPPORTED BEAM WITH POINT MOMENT AT THE CENTER
                # pin at left end
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                # roller at the right end
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 1:3] = 1
                    bcvalues[i, 1:3] = initial_state[i, 1:3]
                # apply moment at the center
                elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 5] = MOMENT_Z
            elif (load_case == 4): # MODE-I COLUMN BUCKLING WITH DISPLACEMENT CONTROL LOADING
                # pin at left end
                if ((x_coord <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bcvalues[i, 0:3] = initial_state[i, 0:3]
                # roller and axial displacement at the right end
                elif ((abs(x_coord - L) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bctypes[i, 0:3] = 1
                    bcvalues[i, 1:3] = initial_state[i, 1:3]
                    bcvalues[i, 0] = initial_state[i, 0] + load_level*DISP_CB
                # perturbation force at the center
                elif ((abs(x_coord - L/2.0) <= SPATIAL_TOLERANCE) and (y_coord <= SPATIAL_TOLERANCE) and (z_coord <= SPATIAL_TOLERANCE)):
                    bcvalues[i, 1] = PERTURB_FORCE

    # # create a VTK directory or clear it
    # if not os.path.isdir("VTK"):
    #     os.mkdir("VTK")
    # else:
    #     for item in os.listdir("VTK"):
    #         os.remove(os.path.join("VTK", item))
    
    # # write the initial displacements
    # PostProcess.write_displacements_forces_vtk("./VTK/output-0", system)

    # incremental computation of the load path
    if (load_case == 4):
        for i in range(0, load_steps):
            print("\nCurrent load step:", i+1,"out of", load_steps, "load steps.")
            load_level = (i+1)/load_steps
            # apply the boundary conditions
            get_BCs(bctypes, bcvalues, load_case, load_level)
            solver.set_boundary_conditions(bctypes, bcvalues)
            # solve the nonlinear static problem and update the system 
            solver.solve(Nmax = 20, tol = 1.0E-03)
            # if ((i+1) % save_step == 0):
            #     output_file = "./VTK/output-" + str(i+1)
            #     PostProcess.write_displacements_forces_vtk(output_file, system)
    else:
        # apply the boundary conditions
        get_BCs(bctypes, bcvalues, load_case)
        solver.set_boundary_conditions(bctypes, bcvalues)
        # solve the nonlinear static problem and update the system 
        solver.solve(Nmax = 10, tol = 1.0E-03)
        # PostProcess.write_displacements_forces_vtk("./VTK/output-1", system)
    
    print(solver.solution)

# run main functions
static_main()