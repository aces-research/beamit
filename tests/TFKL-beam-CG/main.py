from beamit import FunctionSpace
from beamit import Material
from beamit import System
from beamit import Solver
import numpy as np
import copy

def run_analysis(load_case):

    # density of the material
    rho = 7850.0
    # elastic modulus of beam
    E = 2.0E11
    # radius of the beam
    R = 1.0E-01
    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R)

    # length of beam
    L = 10.0
    # number of elements
    Nel = 10
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.TFKLGeometricallyExactFunctionSpace(0, L, Nel, discretization_type = "CG")
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
    load_steps = 100
    save_step = 1

    # applied loads and tolerances
    FORCE_Y = -1.0E04
    MOMENT_Z = 1.0E04
    PERTURB_FORCE = 100.0
    DISP_CB = -0.01
    SPATIAL_TOLERANCE = 1.0E-05
    NUMERICAL_TOLERANCE = 1.0E-05

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
            elif (load_case == 3): # MODE-I COLUMN BUCKLING WITH DISPLACEMENT CONTROL LOADING
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

    # incremental computation of the load path
    if (load_case == 3):
        for i in range(0, load_steps):
            print("\nCurrent load step:", i+1,"out of", load_steps, "load steps.")
            load_level = (i+1)/load_steps
            # apply the boundary conditions
            get_BCs(bctypes, bcvalues, load_case, load_level)
            solver.set_boundary_conditions(bctypes, bcvalues)
            # solve the nonlinear static problem and update the system 
            solver.solve(Nmax = 20, tol = 1.0E-03)
    else:
        # apply the boundary conditions
        get_BCs(bctypes, bcvalues, load_case)
        solver.set_boundary_conditions(bctypes, bcvalues)
        # solve the nonlinear static problem and update the system 
        solver.solve(Nmax = 20, tol = 1.0E-03)

    # asserts for the different load cases
    if (load_case == 0): # SIMPLY SUPPORTED BEAM WITH POINT LOAD
        # maximum deflection at the center
        assert(np.abs(system.state[function_space.N//2, 1] - 
            (FORCE_Y*L**3/(12*E*np.pi*R**4))) < NUMERICAL_TOLERANCE)
    elif (load_case == 1): # CANTILEVER BEAM WITH POINT MOMENT
        # maximum deflection at the right end
        assert(np.abs(system.state[function_space.N-1, 1] - 
            (MOMENT_Z*L**2/(0.50*E*np.pi*R**4))) < NUMERICAL_TOLERANCE)
    elif (load_case == 2): # FIXED-FIXED BEAM WITH POINT LOAD
        # maximum deflection at the center
        assert(np.abs(system.state[function_space.N//2, 1] - 
            (FORCE_Y*L**3/(48*E*np.pi*R**4))) < NUMERICAL_TOLERANCE)
    elif (load_case == 3): # MODE-I COLUMN BUCKLING WITH DISPLACEMENT_CONTROL LOADING
        # axial reaction at the supported end
        EI = (E*np.pi*R**4)/4.0
        Pcr = (np.pi**2*EI)/(L**2)
        axial_reaction = -system.internal_forces[function_space.N-1, 0]
        assert (np.abs(axial_reaction - Pcr) /
                Pcr < NUMERICAL_TOLERANCE*1.0E03)

def test_load_case_0():
    run_analysis(load_case=0)

def test_load_case_1():
    run_analysis(load_case=1)

def test_load_case_2():
    run_analysis(load_case=2)

def test_load_case_3():
    run_analysis(load_case=3)

if __name__ == "__main__":
    test_load_case_0()
    test_load_case_1()
    test_load_case_2()
    test_load_case_3()
