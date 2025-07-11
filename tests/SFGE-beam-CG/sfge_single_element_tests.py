import numpy as np
import quaternion
from beamit import FunctionSpace
from beamit import Material
from beamit.WeakForm.ShearFlexibleGeometricallyExactWeakForm import \
    ShearFlexibleGeometricallyExactWeakFormCG, ShearFlexibleGeometricallyExactWeakFormDG

# density of the material
rho = 1.0
# elastic modulus of beam
E = 1.0
# Poisson's ratio of beam
nu = 0.0
# length of beam
L = 1.0
# beam slenderness ratio
slenderness_ratio = 10.0
# side of the square cross-section
a = L / slenderness_ratio
# number of elements
Nel = 1
# numerical tolerance
NUMERICAL_TOLERANCE = 1.0E-10

# function to compute the skew-symmetric matrix from a vector
def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def test_bulk_internal_variable_updates():

    # physical information (material parameters)
    material = Material.ShearFlexibleMaterial(
        rho, E, nu, A=a**2, I=(a**4)/12, I_minor=(a**4)/12)
    
    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type="CG")
    function_space.discretize()

    # generate the shear flexible weak form
    weak_form = ShearFlexibleGeometricallyExactWeakFormCG(function_space, material)
    
    ###### Case 1: Rigid translation #######
    solution_increment = np.zeros([function_space.N * function_space.dof, 1])
    # rigid translation in X
    solution_increment[0, 0] += 1.0
    solution_increment[6, 0] += 1.0
    # rigid translation in Y
    solution_increment[1, 0] += 5.0
    solution_increment[7, 0] += 5.0
    # rigid translation in Z
    solution_increment[2, 0] += 3.0
    solution_increment[8, 0] += 3.0

    # update the bulk internal variables
    weak_form.update_internal_variables(solution_increment)

    # check the internal variables
    for i in range(0, function_space.E):
        for j in range(0, function_space.Q):
            for k in range(0, function_space.dim):
                # check that the orientation is zero
                assert weak_form.orientation[i, j, k] == 0.0, \
                    f"Orientation at element {i}, quadrature point {j}, dimension {k} is not zero."
                # check that the curvature is zero
                assert weak_form.curvature[i, j, k] == 0.0, \
                    f"Curvature at element {i}, quadrature point {j}, dimension {k} is not zero."
                
    ###### Case 2: Rigid rotation #######
    solution_increment.fill(0.0)
    solution_increment[3, 0] += np.pi / 2.0
    solution_increment[4, 0] += np.pi / 8.0
    solution_increment[5, 0] += np.pi / 4.0
    solution_increment[9, 0] += np.pi / 2.0
    solution_increment[10, 0] += np.pi / 8.0
    solution_increment[11, 0] += np.pi / 4.0

    # update the bulk internal variables
    weak_form.update_internal_variables(solution_increment)

    # check the internal variables
    for i in range(0, function_space.E):
        for j in range(0, function_space.Q):
            # check the orientation
            assert np.isclose(weak_form.orientation[i, j, 0], np.pi / 2.0, atol=NUMERICAL_TOLERANCE), \
                f"Orientation at element {i}, quadrature point {j}, dimension 0 is not pi/2."
            assert np.isclose(weak_form.orientation[i, j, 1], np.pi / 8.0, atol=NUMERICAL_TOLERANCE), \
                f"Orientation at element {i}, quadrature point {j}, dimension 1 is not pi/8."
            assert np.isclose(weak_form.orientation[i, j, 2], np.pi / 4.0, atol=NUMERICAL_TOLERANCE), \
                f"Orientation at element {i}, quadrature point {j}, dimension 2 is not pi/4."
            # check the curvature
            for k in range(0, function_space.dim):
                # check that the curvature is zero
                assert weak_form.curvature[i, j, k] == 0.0, \
                    f"Curvature at element {i}, quadrature point {j}, dimension {k} is not zero."

    ###### Case 3: Constant small curvature #######
    solution_increment.fill(0.0)
    solution_increment[3, 0] += np.pi / 4.0
    solution_increment[4, 0] += np.pi / 8.0
    solution_increment[5, 0] += np.pi / 4.0
    solution_increment[9, 0] -= np.pi / 4.0
    solution_increment[10, 0] -= np.pi / 8.0
    solution_increment[11, 0] -= np.pi / 4.0

    # update the bulk internal variables
    weak_form.update_internal_variables(solution_increment)

    # check the curvatures
    for i in range(0, function_space.Q):
        assert np.isclose(weak_form.curvature[0, i, 0], -np.pi / 2.0, atol=NUMERICAL_TOLERANCE), \
            "Curvature at element 0, quadrature point 0, dimension 0 is not -pi/2."
        assert np.isclose(weak_form.curvature[0, i, 1], -np.pi / 4.0, atol=NUMERICAL_TOLERANCE), \
            "Curvature at element 0, quadrature point 0, dimension 1 is not -pi/4."
        assert np.isclose(weak_form.curvature[0, i, 2], -np.pi / 2.0, atol=NUMERICAL_TOLERANCE), \
            "Curvature at element 0, quadrature point 0, dimension 2 is not -pi/2."
    
def test_residual_CG():

    # physical information (material parameters)
    material = Material.ShearFlexibleMaterial(
        rho, E, nu, A=a**2, I=(a**4)/12, I_minor=(a**4)/12)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type="CG")
    function_space.discretize()

    # generate the shear flexible weak form
    weak_form = ShearFlexibleGeometricallyExactWeakFormCG(
        function_space, material)
    
    # generate a solution increment
    solution = np.zeros([function_space.N * function_space.dof, 1])
    solution[0, 0] = 1.0
    solution[1, 0] = 5.0
    solution[2, 0] = 3.0
    solution[3, 0] = np.pi / 4.0
    solution[4, 0] = np.pi / 8.0
    solution[5, 0] = np.pi / 16.0
    solution[6, 0] = 10.0
    solution[7, 0] = 6.0
    solution[8, 0] = 8.0
    solution[9, 0] = np.pi / 16.0
    solution[10, 0] = np.pi / 8.0
    solution[11, 0] = np.pi / 32.0

    # update the bulk internal variables
    weak_form.update_internal_variables(solution)

    # compute the residual
    residual_computed = np.zeros([function_space.N * function_space.dof, 1])
    weak_form.compute_system_residual(
        residual_computed, solution, None, update_internal=False)

    # calculate the expected residual
    residual_expected = np.zeros([function_space.N * function_space.dof, 1])
    quad_points, quad_weights = np.polynomial.legendre.leggauss(
        function_space.Q)
    for i in range(0, quad_points.shape[0]):
        N1 = 0.50*(1.0 - quad_points[i])
        N2 = 0.50*(1.0 + quad_points[i])
        N1_xi = -0.50
        N2_xi = 0.50
        shapes = np.array([[N1, 0.0, 0.0, N2, 0.0, 0.0],
                           [0.0, N1, 0.0, 0.0, N2, 0.0],
                           [0.0, 0.0, N1, 0.0, 0.0, N2]])
        shape_first_gradients = np.array([[N1_xi, 0.0, 0.0, N2_xi, 0.0, 0.0],
                                          [0.0, N1_xi, 0.0, 0.0, N2_xi, 0.0],
                                          [0.0, 0.0, N1_xi, 0.0, 0.0, N2_xi]])
        # expected internal forces
        rp_element = (np.array([[10.0], [6.0], [8.0]]) - np.array([[1.0], [5.0], [3.0]])) / L
        theta = N1 * (np.array([np.pi / 4.0, np.pi / 8.0, np.pi / 16.0])) + \
            N2 * (np.array([np.pi / 16.0, np.pi / 8.0, np.pi / 32.0]))
        element_orientations_tensor = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(theta))
        C_F = np.zeros([3, 3])
        C_F[0, 0] = material.E * material.A
        C_F[1, 1] = material.G * material.A_red
        C_F[2, 2] = material.G * material.A_red
        C_F_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_F, np.transpose(element_orientations_tensor)))
        element_strains = rp_element - \
            np.matmul(element_orientations_tensor, np.array([[1.0], [0.0], [0.0]]))
        element_internal_forces = np.matmul(C_F_transformed, element_strains)
        residual_forces = np.matmul(np.transpose(
            shape_first_gradients), element_internal_forces)*quad_weights[i]
        residual_expected[0:3, 0] -= residual_forces[0:3, 0]
        residual_expected[6:9, 0] -= residual_forces[3:6, 0]
        # expected internal moments
        C_M = np.zeros([3, 3])
        C_M[0, 0] = material.G * material.I_T
        C_M[1, 1] = material.E * material.I
        C_M[2, 2] = material.E * material.I_minor
        C_M_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_M, np.transpose(element_orientations_tensor)))
        dtheta_prime = (np.array([[np.pi / 16.0], [np.pi / 8.0], [np.pi / 32.0]]) - \
            np.array([[np.pi / 4.0], [np.pi / 8.0], [np.pi / 16.0]])) / L
        theta_L2 = np.linalg.norm(theta)
        T_matrix = np.sin(theta_L2) / theta_L2 * np.eye(3) + \
            (1 - np.cos(theta_L2)) / (theta_L2**2) * skew(theta) + \
            (theta_L2 - np.sin(theta_L2)) / (theta_L2**3) * np.outer(theta, theta)
        curvature = np.matmul(T_matrix, dtheta_prime)
        element_internal_moments = np.matmul(C_M_transformed, curvature)
        residual_moments = np.matmul(np.transpose(
            shape_first_gradients), element_internal_moments)*quad_weights[i]
        rp_cross_internal_fores = np.cross(rp_element, element_internal_forces, axis=0)
        residual_moments -= np.matmul(np.transpose(shapes),
                                      rp_cross_internal_fores)*0.50*L*quad_weights[i]
        residual_expected[3:6, 0] -= residual_moments[0:3, 0]
        residual_expected[9:12, 0] -= residual_moments[3:6, 0]

    # check the residual
    assert np.allclose(residual_computed, residual_expected, atol=NUMERICAL_TOLERANCE), \
        "Residual computed does not match the expected residual."

def test_stiffness_CG():

    # physical information (material parameters)
    material = Material.ShearFlexibleMaterial(
        rho, E, nu, A=a**2, I=(a**4)/12, I_minor=(a**4)/12)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, L, Nel, discretization_type="CG")
    function_space.discretize()

    # generate the shear flexible weak form
    weak_form = ShearFlexibleGeometricallyExactWeakFormCG(
        function_space, material)

    # generate a solution increment
    solution = np.zeros([function_space.N * function_space.dof, 1])
    solution[0, 0] = 1.0
    solution[1, 0] = 5.0
    solution[2, 0] = 3.0
    solution[3, 0] = np.pi / 4.0
    solution[4, 0] = np.pi / 8.0
    solution[5, 0] = np.pi / 16.0
    solution[6, 0] = 10.0
    solution[7, 0] = 6.0
    solution[8, 0] = 8.0
    solution[9, 0] = np.pi / 16.0
    solution[10, 0] = np.pi / 8.0
    solution[11, 0] = np.pi / 32.0

    # update the bulk internal variables
    weak_form.update_internal_variables(solution)

    # compute the stiffness matrix
    stiffness_computed = np.zeros(
        [function_space.N * function_space.dof, function_space.N * function_space.dof])
    weak_form.compute_system_stiffness(stiffness_computed, solution, nodal_loads=None, 
                                       element_loads=None)
    
    # calculate the expected stiffness matrix
    stiffness_expected = np.zeros(
        [function_space.N * function_space.dof, function_space.N * function_space.dof])
    quad_points, quad_weights = np.polynomial.legendre.leggauss(
        function_space.Q)
    translation_dofs = np.array([0, 1, 2, 6, 7, 8])
    rotation_dofs = np.array([3, 4, 5, 9, 10, 11])
    for i in range(0, quad_points.shape[0]):
        N1 = 0.50*(1.0 - quad_points[i])
        N2 = 0.50*(1.0 + quad_points[i])
        N1_xi = -0.50
        N2_xi = 0.50
        shapes = np.array([[N1, 0.0, 0.0, N2, 0.0, 0.0],
                           [0.0, N1, 0.0, 0.0, N2, 0.0],
                           [0.0, 0.0, N1, 0.0, 0.0, N2]])
        shape_first_gradients = np.array([[N1_xi, 0.0, 0.0, N2_xi, 0.0, 0.0],
                                          [0.0, N1_xi, 0.0, 0.0, N2_xi, 0.0],
                                          [0.0, 0.0, N1_xi, 0.0, 0.0, N2_xi]])*(2.0/L)
        # expected internal forces
        rp_element = (np.array([[10.0], [6.0], [8.0]]) -
                      np.array([[1.0], [5.0], [3.0]])) / L
        theta = N1 * (np.array([np.pi / 4.0, np.pi / 8.0, np.pi / 16.0])) + \
            N2 * (np.array([np.pi / 16.0, np.pi / 8.0, np.pi / 32.0]))
        element_orientations_tensor = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(theta))
        C_F = np.zeros([3, 3])
        C_F[0, 0] = material.E * material.A
        C_F[1, 1] = material.G * material.A_red
        C_F[2, 2] = material.G * material.A_red
        C_F_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_F, np.transpose(element_orientations_tensor)))
        element_strains = rp_element - \
            np.matmul(element_orientations_tensor,
                      np.array([[1.0], [0.0], [0.0]]))
        element_internal_forces = np.matmul(C_F_transformed, element_strains)
        # expected internal moments
        C_M = np.zeros([3, 3])
        C_M[0, 0] = material.G * material.I_T
        C_M[1, 1] = material.E * material.I
        C_M[2, 2] = material.E * material.I_minor
        C_M_transformed = np.matmul(element_orientations_tensor, np.matmul(
            C_M, np.transpose(element_orientations_tensor)))
        dtheta_prime = (np.array([[np.pi / 16.0], [np.pi / 8.0], [np.pi / 32.0]]) -
                        np.array([[np.pi / 4.0], [np.pi / 8.0], [np.pi / 16.0]])) / L
        theta_L2 = np.linalg.norm(theta)
        T_matrix = np.sin(theta_L2) / theta_L2 * np.eye(3) + \
            (1 - np.cos(theta_L2)) / (theta_L2**2) * skew(theta) + \
            (theta_L2 - np.sin(theta_L2)) / \
            (theta_L2**3) * np.outer(theta, theta)
        curvature = np.matmul(T_matrix, dtheta_prime)
        element_internal_moments = np.matmul(C_M_transformed, curvature)
        # the material part of the stiffness matrix
        stiffness_expected[np.ix_(translation_dofs, translation_dofs)] += \
            np.matmul(np.transpose(shape_first_gradients), np.matmul(
                C_F_transformed, shape_first_gradients)) * (L/2) * quad_weights[i]
        stiffness_expected[np.ix_(translation_dofs, rotation_dofs)] += \
            np.matmul(np.transpose(shape_first_gradients), np.matmul(
                C_F_transformed, np.matmul(skew(rp_element[..., 0]), shapes))) * (L/2) * quad_weights[i]
        stiffness_expected[np.ix_(rotation_dofs, translation_dofs)] -= \
            np.matmul(np.transpose(shapes), np.matmul(skew(rp_element[..., 0]), np.matmul(
                C_F_transformed, shape_first_gradients))) * (L/2) * quad_weights[i]
        stiffness_expected[np.ix_(rotation_dofs, rotation_dofs)] += \
            np.matmul(np.transpose(shape_first_gradients), np.matmul(
                C_M_transformed, shape_first_gradients)) * (L/2) * quad_weights[i]
        stiffness_expected[np.ix_(rotation_dofs, rotation_dofs)] -= \
            np.matmul(np.matmul(np.transpose(shapes), skew(rp_element[..., 0])), np.matmul(
            C_F_transformed, np.matmul(skew(rp_element[..., 0]), shapes))) * (L/2) * quad_weights[i]
        # the geometric part of the stiffness matrix
        stiffness_expected[np.ix_(translation_dofs, rotation_dofs)] -= \
            np.matmul(np.transpose(shape_first_gradients), np.matmul(skew(
                element_internal_forces[..., 0]), shapes)) * (L/2) * quad_weights[i]
        stiffness_expected[np.ix_(rotation_dofs, translation_dofs)] += \
            np.matmul(np.transpose(shapes), np.matmul(skew(
                element_internal_forces[..., 0]), shape_first_gradients)) * (L/2) * quad_weights[i]
        stiffness_expected[np.ix_(rotation_dofs, rotation_dofs)] -= \
            np.matmul(np.transpose(shape_first_gradients), np.matmul(skew(
                element_internal_moments[..., 0]), shapes)) * (L/2) * quad_weights[i]
        term4_pre_factor = np.outer(element_internal_forces, rp_element) - \
            np.dot(element_internal_forces[..., 0],
                   rp_element[..., 0])*np.eye(3)
        stiffness_expected[np.ix_(rotation_dofs, rotation_dofs)] += \
            np.matmul(np.transpose(shapes), np.matmul(
                term4_pre_factor, shapes)) * (L/2) * quad_weights[i]

    # check the stiffness matrix
    assert np.allclose(stiffness_computed, stiffness_expected, atol=NUMERICAL_TOLERANCE), \
        "Stiffness matrix computed does not match the expected stiffness matrix."

def test_residual_DG():

    # physical information (material parameters)
    material = Material.ShearFlexibleMaterial(
        rho, E, nu, A=a**2, I=(a**4)/12, I_minor=(a**4)/12)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.ShearFlexibleGeometricallyExactFunctionSpace(
        0, 2.0*L, 2*Nel, discretization_type="DG")
    function_space.discretize()

    # generate the shear flexible weak form
    weak_form = ShearFlexibleGeometricallyExactWeakFormDG(
        function_space, material, betaP=10.0, betaT=10.0)
    
    # generate a solution increment
    solution = np.zeros([function_space.N * function_space.dof, 1])
    solution[0, 0] = 1.0
    solution[1, 0] = 5.0
    solution[2, 0] = 3.0
    solution[3, 0] = np.pi / 4.0
    solution[4, 0] = np.pi / 8.0
    solution[5, 0] = np.pi / 16.0
    solution[6, 0] = 10.0
    solution[7, 0] = 6.0
    solution[8, 0] = 8.0
    solution[9, 0] = np.pi / 16.0
    solution[10, 0] = np.pi / 8.0
    solution[11, 0] = np.pi / 32.0
    solution[12, 0] = 2.0
    solution[13, 0] = 10.0
    solution[14, 0] = 8.0
    solution[15, 0] = np.pi / 6.0
    solution[16, 0] = np.pi / 4.0
    solution[17, 0] = np.pi / 8.0
    solution[18, 0] = 6.0
    solution[19, 0] = 10.0
    solution[20, 0] = 4.0
    solution[21, 0] = np.pi / 8.0
    solution[22, 0] = np.pi / 3.0
    solution[23, 0] = np.pi / 8.0

    # update the bulk internal variables
    weak_form.update_internal_variables(solution)

    # compute the residual
    residual_computed = np.zeros([function_space.N * function_space.dof, 1])
    weak_form.compute_system_residual(
        residual_computed, solution, None, update_internal=False)
    
    # calculate the expected residual
    residual_expected = np.zeros([function_space.N * function_space.dof, 1])
    quad_points, quad_weights = np.polynomial.legendre.leggauss(
        function_space.Q)
    # assemble the bulk residual
    dofspel = function_space.dof * function_space.npel
    for e in range(0, 2):
        for i in range(0, quad_points.shape[0]):
            N1 = 0.50*(1.0 - quad_points[i])
            N2 = 0.50*(1.0 + quad_points[i])
            N1_xi = -0.50
            N2_xi = 0.50
            shapes = np.array([[N1, 0.0, 0.0, N2, 0.0, 0.0],
                               [0.0, N1, 0.0, 0.0, N2, 0.0],
                               [0.0, 0.0, N1, 0.0, 0.0, N2]])
            shape_first_gradients = np.array([[N1_xi, 0.0, 0.0, N2_xi, 0.0, 0.0],
                                            [0.0, N1_xi, 0.0, 0.0, N2_xi, 0.0],
                                            [0.0, 0.0, N1_xi, 0.0, 0.0, N2_xi]])*(2.0/L)
            # expected internal forces
            if (e == 0):
                rp_element = (np.array([[10.0], [6.0], [8.0]]) -
                            np.array([[1.0], [5.0], [3.0]])) / L
                theta = N1 * (np.array([np.pi / 4.0, np.pi / 8.0, np.pi / 16.0])) + \
                    N2 * (np.array([np.pi / 16.0, np.pi / 8.0, np.pi / 32.0]))
            elif (e == 1):
                rp_element = (np.array([[6.0], [10.0], [4.0]]) -
                            np.array([[2.0], [10.0], [8.0]])) / L
                theta = N1 * (np.array([np.pi / 6.0, np.pi / 4.0, np.pi / 8.0])) + \
                    N2 * (np.array([np.pi / 8.0, np.pi / 3.0, np.pi / 8.0]))
            element_orientations_tensor = quaternion.as_rotation_matrix(
                quaternion.from_rotation_vector(theta))
            C_F = np.zeros([3, 3])
            C_F[0, 0] = material.E * material.A
            C_F[1, 1] = material.G * material.A_red
            C_F[2, 2] = material.G * material.A_red
            C_F_transformed = np.matmul(element_orientations_tensor, np.matmul(
                C_F, np.transpose(element_orientations_tensor)))
            element_strains = rp_element - \
                np.matmul(element_orientations_tensor,
                        np.array([[1.0], [0.0], [0.0]]))
            element_internal_forces = np.matmul(C_F_transformed, element_strains)
            residual_forces = np.matmul(np.transpose(
                shape_first_gradients), element_internal_forces)*0.50*L*quad_weights[i]
            residual_expected[0+e*dofspel:3+e *
                              dofspel, 0] -= residual_forces[0:3, 0]
            residual_expected[6+e*dofspel:9+e *
                              dofspel, 0] -= residual_forces[3:6, 0]
            # expected internal moments
            C_M = np.zeros([3, 3])
            C_M[0, 0] = material.G * material.I_T
            C_M[1, 1] = material.E * material.I
            C_M[2, 2] = material.E * material.I_minor
            C_M_transformed = np.matmul(element_orientations_tensor, np.matmul(
                C_M, np.transpose(element_orientations_tensor)))
            if (e == 0):
                dtheta_prime = (np.array([[np.pi / 16.0], [np.pi / 8.0], [np.pi / 32.0]]) -
                                np.array([[np.pi / 4.0], [np.pi / 8.0], [np.pi / 16.0]])) / L
            elif (e == 1):
                dtheta_prime = (np.array([[np.pi / 8.0], [np.pi / 3.0], [np.pi / 8.0]]) -
                                np.array([[np.pi / 6.0], [np.pi / 4.0], [np.pi / 8.0]])) / L
            theta_L2 = np.linalg.norm(theta)
            T_matrix = np.sin(theta_L2) / theta_L2 * np.eye(3) + \
                (1 - np.cos(theta_L2)) / (theta_L2**2) * skew(theta) + \
                (theta_L2 - np.sin(theta_L2)) / \
                (theta_L2**3) * np.outer(theta, theta)
            curvature = np.matmul(T_matrix, dtheta_prime)
            element_internal_moments = np.matmul(C_M_transformed, curvature)
            residual_moments = np.matmul(np.transpose(
                shape_first_gradients), element_internal_moments)*0.50*L*quad_weights[i]
            rp_cross_internal_fores = np.cross(
                rp_element, element_internal_forces, axis=0)
            residual_moments -= np.matmul(np.transpose(shapes),
                                        rp_cross_internal_fores)*0.50*L*quad_weights[i]
            residual_expected[3+e*dofspel:6+e *
                              dofspel, 0] -= residual_moments[0:3, 0]
            residual_expected[9+e*dofspel:12+e*dofspel,
                              0] -= residual_moments[3:6, 0]

    # assemble the interface residual
    N1_xi = -0.50
    N2_xi = 0.50
    translation_dofs = np.array([0, 1, 2, 6, 7, 8])
    rotation_dofs = np.array([3, 4, 5, 9, 10, 11])
    N_left_interface = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    N_right_interface = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    Np_left_interface = np.array([[N1_xi, 0.0, 0.0, N2_xi, 0.0, 0.0],
                                  [0.0, N1_xi, 0.0, 0.0, N2_xi, 0.0],
                                  [0.0, 0.0, N1_xi, 0.0, 0.0, N2_xi]])*(2.0/L)
    Np_right_interface = np.array([[N1_xi, 0.0, 0.0, N2_xi, 0.0, 0.0],
                                  [0.0, N1_xi, 0.0, 0.0, N2_xi, 0.0],
                                  [0.0, 0.0, N1_xi, 0.0, 0.0, N2_xi]])*(2.0/L)
    rp_left_interface = (np.array([[10.0], [6.0], [8.0]]) -
                         np.array([[1.0], [5.0], [3.0]])) / L
    rp_right_interface = (np.array([[6.0], [10.0], [4.0]]) -
                          np.array([[2.0], [10.0], [8.0]])) / L
    theta_left_interface = np.array([np.pi / 16.0, np.pi / 8.0, np.pi / 32.0])
    theta_right_interface = np.array([np.pi / 6.0, np.pi / 4.0, np.pi / 8.0])
    dtheta_left_interface = (np.array([[np.pi / 16.0], [np.pi / 8.0], [np.pi / 32.0]]) -
                             np.array([[np.pi / 4.0], [np.pi / 8.0], [np.pi / 16.0]])) / L
    dtheta_right_interface = (np.array([[np.pi / 8.0], [np.pi / 3.0], [np.pi / 8.0]]) -
                              np.array([[np.pi / 6.0], [np.pi / 4.0], [np.pi / 8.0]])) / L
    element_orientations_tensor_left = quaternion.as_rotation_matrix(
        quaternion.from_rotation_vector(theta_left_interface))
    element_orientations_tensor_right = quaternion.as_rotation_matrix(
        quaternion.from_rotation_vector(theta_right_interface))
    C_F = np.zeros([3, 3])
    C_F[0, 0] = material.E * material.A
    C_F[1, 1] = material.G * material.A_red
    C_F[2, 2] = material.G * material.A_red
    C_M = np.zeros([3, 3])
    C_M[0, 0] = material.G * material.I_T
    C_M[1, 1] = material.E * material.I
    C_M[2, 2] = material.E * material.I_minor
    C_F_transformed_left = np.matmul(element_orientations_tensor_left, np.matmul(
        C_F, np.transpose(element_orientations_tensor_left)))
    C_F_transformed_right = np.matmul(element_orientations_tensor_right, np.matmul(
        C_F, np.transpose(element_orientations_tensor_right)))
    C_M_transformed_left = np.matmul(element_orientations_tensor_left, np.matmul(
        C_M, np.transpose(element_orientations_tensor_left)))
    C_M_transformed_right = np.matmul(element_orientations_tensor_right, np.matmul(
        C_M, np.transpose(element_orientations_tensor_right)))
    element_strains_left = rp_left_interface - \
        np.matmul(element_orientations_tensor_left, np.array([[1.0], [0.0], [0.0]]))
    element_strains_right = rp_right_interface - \
        np.matmul(element_orientations_tensor_right, np.array([[1.0], [0.0], [0.0]]))
    element_internal_forces_left = np.matmul(C_F_transformed_left, element_strains_left)
    element_internal_forces_right = np.matmul(C_F_transformed_right, element_strains_right)
    theta_L2_left = np.linalg.norm(theta_left_interface)
    theta_L2_right = np.linalg.norm(theta_right_interface)
    T_matrix_left = np.sin(theta_L2_left) / theta_L2_left * np.eye(3) + \
        (1 - np.cos(theta_L2_left)) / (theta_L2_left**2) * skew(theta_left_interface) + \
        (theta_L2_left - np.sin(theta_L2_left)) / (theta_L2_left**3) * \
        np.outer(theta_left_interface, theta_left_interface)
    T_matrix_right = np.sin(theta_L2_right) / theta_L2_right * np.eye(3) + \
        (1 - np.cos(theta_L2_right)) / (theta_L2_right**2) * skew(theta_right_interface) + \
        (theta_L2_right - np.sin(theta_L2_right)) / (theta_L2_right**3) * \
        np.outer(theta_right_interface, theta_right_interface)
    curvature_left = np.matmul(T_matrix_left, dtheta_left_interface)
    curvature_right = np.matmul(T_matrix_right, dtheta_right_interface)
    element_internal_moments_left = np.matmul(C_M_transformed_left, curvature_left)
    element_internal_moments_right = np.matmul(C_M_transformed_right, curvature_right)
    average_internal_forces = 0.5 * \
        (element_internal_forces_left + element_internal_forces_right)
    average_internal_moments = 0.5 * \
        (element_internal_moments_left + element_internal_moments_right)
    # flux terms
    residual_expected[translation_dofs] += np.matmul(
        np.transpose(N_left_interface), average_internal_forces)
    residual_expected[rotation_dofs] += np.matmul(
        np.transpose(N_left_interface), average_internal_moments)
    residual_expected[translation_dofs + dofspel] -= np.matmul(
        np.transpose(N_right_interface), average_internal_forces)
    residual_expected[rotation_dofs + dofspel] -= np.matmul(
        np.transpose(N_right_interface), average_internal_moments)
    # extra terms
    residual_expected[translation_dofs] -= 0.50 * np.matmul(
        np.transpose(Np_left_interface), np.matmul(C_F_transformed_left, 
                                                   weak_form.dof_jumps_boundaries[0:1, 6:9].T))
    residual_expected[translation_dofs + dofspel] -= 0.50 * np.matmul(
        np.transpose(Np_right_interface), np.matmul(C_F_transformed_right, 
                                                    weak_form.dof_jumps_boundaries[0:1, 6:9].T))
    residual_expected[rotation_dofs] -= 0.50 * np.matmul(
        np.transpose(Np_left_interface), np.matmul(C_M_transformed_left,
                                                   weak_form.dof_jumps_boundaries[0:1, 9:12].T))
    residual_expected[rotation_dofs + dofspel] -= 0.50 * np.matmul(
        np.transpose(Np_right_interface), np.matmul(C_M_transformed_right, 
                                                    weak_form.dof_jumps_boundaries[0:1, 9:12].T))
    residual_expected[rotation_dofs] += 0.50 * np.matmul(
        np.transpose(N_left_interface), np.cross(
            weak_form.dof_jumps_boundaries[0:1, 6:9].T, element_internal_forces_left, axis=0))
    residual_expected[rotation_dofs + dofspel] += 0.50 * np.matmul(
        np.transpose(N_right_interface), np.cross(
            weak_form.dof_jumps_boundaries[0:1, 6:9].T, element_internal_forces_right, axis=0))
    residual_expected[rotation_dofs] += 0.50 * np.matmul(
        np.transpose(N_left_interface), np.cross(
            rp_left_interface, np.matmul(C_F_transformed_left, weak_form.dof_jumps_boundaries[0:1, 6:9].T), axis=0))
    residual_expected[rotation_dofs + dofspel] += 0.50 * np.matmul(
        np.transpose(N_right_interface), np.cross(
            rp_right_interface, np.matmul(C_F_transformed_right, weak_form.dof_jumps_boundaries[0:1, 6:9].T), axis=0))
    # penalty terms
    penalty_forces = 10.0*((material.E*material.A)/L) * \
        weak_form.dof_jumps_boundaries[0:1, 6:9].T
    penalty_moments = 10.0*((material.E*material.I)/L) * \
        weak_form.dof_jumps_boundaries[0:1, 9:12].T
    residual_expected[translation_dofs] += np.matmul(
        np.transpose(N_left_interface), penalty_forces)
    residual_expected[rotation_dofs] += np.matmul(
        np.transpose(N_left_interface), penalty_moments)
    residual_expected[translation_dofs + dofspel] -= np.matmul(
        np.transpose(N_right_interface), penalty_forces)
    residual_expected[rotation_dofs + dofspel] -= np.matmul(
        np.transpose(N_right_interface), penalty_moments)

    assert np.allclose(residual_computed, residual_expected, atol=NUMERICAL_TOLERANCE), \
        "Residual computed does not match the expected residual."

if __name__ == "__main__":

    # run the tests
    test_bulk_internal_variable_updates()
    test_residual_CG()
    test_stiffness_CG()
    test_residual_DG()
