import numpy as np
from beamit import FunctionSpace, Material
from beamit import Material
from beamit import System

# density of the material
rho = 1000.0
# elastic modulus of beam
E = 1.0E12
# the radius of the beam
R = 1.0E-03
# length of beam
L = 0.1
# number of elements
Nel = 1
# tolerance
NUMERICAL_TOLERANCE = 1.0E-08

def test_stiffness_and_residual():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0.0, L, Nel, discretization_type = "CG")
    function_space.discretize()

    # a system binding the function_space (math) and the material (physics) 
    system = System.System(function_space, material)

    # test the element stiffness matrix
    computed_stiffness_matrix = \
        system.weak_form.compute_element_internal_stiffness(np.zeros([function_space.npel*function_space.dof, 1]))
    bending_stiffness_matrix = np.array([[((12.0*E*material.I)/L**3.0), ((6.0*E*material.I)/L**2.0), \
                                       -((12.0*E*material.I)/L**3.0), ((6.0*E*material.I)/L**2.0)], \
                                        [((6.0*E*material.I)/L**2.0), ((4.0*E*material.I)/L), \
                                         -((6.0*E*material.I)/L**2.0), ((2.0*E*material.I)/L)], \
                                        [-((12.0*E*material.I)/L**3.0), -((6.0*E*material.I)/L**2.0), \
                                         ((12.0*E*material.I)/L**3.0), -((6.0*E*material.I)/L**2.0)], \
                                        [((6.0*E*material.I)/L**2.0), ((2.0*E*material.I)/L), \
                                         -((6.0*E*material.I)/L**2.0), ((4.0*E*material.I)/L)]])
    axial_stiffness_matrix = np.array([[(E*material.A)/L, -(E*material.A)/L], \
                                       [-(E*material.A)/L, (E*material.A)/L]])
    actual_stiffness_matrix = np.zeros([function_space.npel*function_space.dof, \
                                            function_space.npel*function_space.dof])
    actual_stiffness_matrix[np.ix_([0, 3], [0, 3])] = axial_stiffness_matrix
    actual_stiffness_matrix[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = bending_stiffness_matrix

    assert np.linalg.norm(computed_stiffness_matrix - actual_stiffness_matrix) < NUMERICAL_TOLERANCE, \
            f"Euler-Bernoulli beam stiffness matrix test failed."

    # test the internal element residual
    random_solution = np.random.rand(function_space.npel*function_space.dof, 1)
    computed_internal_residual = \
        system.weak_form.compute_element_internal_forces(random_solution)
    assert np.linalg.norm(computed_internal_residual - \
                          np.matmul(actual_stiffness_matrix, random_solution)) < NUMERICAL_TOLERANCE, \
            f"Euler-Bernoulli beam internal residual test failed."

def test_consistent_mass():

    # physical information (material parameters)
    material = Material.TFKLMaterial(rho, E, R=R)

    # geometric information (domain, no. of elements)
    function_space = FunctionSpace.EulerBernoulliFunctionSpace(0.0, L, Nel, discretization_type = "CG")
    function_space.discretize()

    # a system binding the function_space (math) and the material (physics)
    system = System.System(function_space, material)

    computed_mass_matrix = np.zeros([system.nequations, system.nequations])
    system.weak_form.compute_system_mass(computed_mass_matrix, \
                        np.zeros([function_space.npel*function_space.dof, 1]), \
                        use_rotational_mass=False, lump=False)

    # test the element mass matrix
    bending_mass_matrix = ((rho*material.A*L)/420.0) * \
                            np.array([[156.0, 22.0*L, 54.0, -13.0*L], \
                                      [22.0*L, 4.0*L*L, 13.0*L, -3.0*L*L], 
                                      [54.0, 13.0*L, 156.0, -22.0*L], 
                                      [-13.0*L, -3.0*L*L, -22.0*L, 4.0*L*L]])
    axial_mass_matrix = ((rho*material.A*L)/420.0) * \
                            np.array([[140.0, 70.0], \
                                      [70.0, 140.0]])
    actual_mass_matrix = np.zeros([function_space.npel*function_space.dof, \
                                            function_space.npel*function_space.dof])
    actual_mass_matrix[np.ix_([0, 3], [0, 3])] = axial_mass_matrix
    actual_mass_matrix[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = bending_mass_matrix

    assert np.linalg.norm(computed_mass_matrix - actual_mass_matrix) < NUMERICAL_TOLERANCE, \
            f"Euler-Bernoulli beam mass matrix test failed."
    
    computed_mass_matrix = np.zeros([system.nequations, system.nequations])
    system.weak_form.compute_system_mass(computed_mass_matrix, \
                        np.zeros([function_space.npel*function_space.dof, 1]), \
                        use_rotational_mass=False, lump=True)

if __name__ == "__main__":

    # run the tests
    test_stiffness_and_residual()
    test_consistent_mass()