import sys
import numpy as np
import matplotlib.pyplot as plt

def compute_shape_function_coefficients(nodal_positions):
    number_nodes = nodal_positions.shape[0]
    shape_function_coefficients = np.zeros([number_nodes, number_nodes])
    # construct the left hand side shape solve matrix
    shape_solve_lhs = np.ones([number_nodes, number_nodes])
    for i in range(number_nodes):
        for j in range(number_nodes):
            shape_solve_lhs[i:i+1, j:j+1] = nodal_positions[i]**j
    # get the shape function coefficients
    for i in range(number_nodes):
        shape_solve_rhs = np.zeros([number_nodes, 1])
        shape_solve_rhs[i] = 1.0
        shape_function_coefficients[i:i+1, :] = np.transpose(np.linalg.solve(shape_solve_lhs, shape_solve_rhs))
    return shape_function_coefficients

def evaluate_step_function_DG_gradients(element_length, polynomial_order, number_plot_points=10):
    number_nodes = polynomial_order + 1
    ####### COMPUTING THE MASS MATRIX OF THE ELEMENTS #######
    parametric_nodal_positions = np.linspace(-1.0, 1.0, number_nodes)
    parametric_shape_function_coefficients = compute_shape_function_coefficients(parametric_nodal_positions)
    nodal_positions_left_element = np.linspace(0.0, element_length, number_nodes)
    shape_function_coefficients_left_element = compute_shape_function_coefficients(nodal_positions_left_element)
    nodal_positions_right_element = np.linspace(element_length, 2.0*element_length, number_nodes)
    shape_function_coefficients_right_element = compute_shape_function_coefficients(nodal_positions_right_element)
    # quadrature points and weights
    number_quadrature_points = int(np.ceil(0.50*(2*polynomial_order + 1.0)))
    quadrature_points, quadrature_weights = np.polynomial.legendre.leggauss(number_quadrature_points)
    # elemental mass matrix (in the context of left element)
    # the mass matrices of left and right elements will be identical!
    element_mass = np.zeros([number_nodes, number_nodes])
    # evaluate jacobians at the quadrature points
    jacobian_quadrature = np.zeros([number_quadrature_points, 1])
    for q in range(number_quadrature_points):
        for i in range(number_nodes):
            jacobian_quadrature[q:q+1, 0:1] += np.sum(parametric_shape_function_coefficients[i:i+1, 1:]*(np.arange(1, number_nodes)[np.newaxis,:])*(np.power(quadrature_points[q], np.arange(0, number_nodes-1))[np.newaxis,:])*nodal_positions_left_element[i], axis=1, keepdims=True)
    # compute the mass matrix
    for i in range(number_nodes):
        for j in range(number_nodes):
            for q in range(number_quadrature_points):
                shape_quadrature_value_i = np.sum(parametric_shape_function_coefficients[i:i+1, :]*(np.power(quadrature_points[q], np.arange(0, number_nodes))[np.newaxis,:]), axis=1, keepdims=True)
                shape_quadrature_value_j = np.sum(parametric_shape_function_coefficients[j:j+1, :]*(np.power(quadrature_points[q], np.arange(0, number_nodes))[np.newaxis,:]), axis=1, keepdims=True)
                element_mass[i:i+1, j:j+1] += shape_quadrature_value_i*shape_quadrature_value_j*jacobian_quadrature[q:q+1, 0:1]*quadrature_weights[q]
    ####### SOLVE FOR THE DG GRADIENT COEFFICIENTS #######
    left_element_gradient_rhs = np.zeros([number_nodes, 1])
    left_element_gradient_rhs[-1:,0:1] = 0.50 # the inter-element boundary trace!
    right_element_gradient_rhs = np.zeros([number_nodes, 1])
    right_element_gradient_rhs[0:1,0:1] = 0.50 # the inter-element boundary trace!
    left_element_gradient_coefficients = np.linalg.solve(element_mass, left_element_gradient_rhs)
    right_element_gradient_coefficients = np.linalg.solve(element_mass, right_element_gradient_rhs)
    ####### EVALUATE DG GRADIENTS AT THE PLOT POINTS #######
    plot_positions_left_element = np.linspace(0.0, element_length, number_plot_points)
    plot_positions_right_element = np.linspace(element_length, 2.0*element_length, number_plot_points)
    DG_gradients_left_element = np.zeros([number_plot_points, 1])
    DG_gradients_right_element = np.zeros([number_plot_points, 1])
    for p in range(number_plot_points):
        shape_values_left_element = np.zeros([number_nodes, 1])
        shape_values_right_element = np.zeros([number_nodes, 1])
        for i in range(number_nodes):
            shape_values_left_element[i:i+1, 0:1] = np.sum(shape_function_coefficients_left_element[i:i+1, :]*(np.power(plot_positions_left_element[p], np.arange(0, number_nodes))[np.newaxis,:]), axis=1, keepdims=True)
            shape_values_right_element[i:i+1, 0:1] = np.sum(shape_function_coefficients_right_element[i:i+1, :]*(np.power(plot_positions_right_element[p], np.arange(0, number_nodes))[np.newaxis,:]), axis=1, keepdims=True)
        DG_gradients_left_element[p:p+1, 0:1] = np.sum(left_element_gradient_coefficients*shape_values_left_element, axis=0, keepdims=True)
        DG_gradients_right_element[p:p+1, 0:1] = np.sum(right_element_gradient_coefficients*shape_values_right_element, axis=0, keepdims=True)
    return np.concatenate((plot_positions_left_element, plot_positions_right_element), axis=0), np.concatenate((DG_gradients_left_element, DG_gradients_right_element), axis=0)

if __name__ == '__main__':
    # element length
    Le = 0.50
    # expect lists of polynomial orders and number of plot points
    polynomial_orders = eval(sys.argv[1])
    fig = plt.figure(figsize=(8.32, 6.24))
    plt.rc("font", size=20)
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    for i in range(len(polynomial_orders)):
        plot_positions, DG_gradients = evaluate_step_function_DG_gradients(Le, polynomial_orders[i], number_plot_points=100)
        plt.plot(plot_positions, DG_gradients, label=r"$k = "+str(polynomial_orders[i])+r"$", linewidth=2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\partial^{DG}_xu$")
    plt.legend()
    plt.show()
    fig.savefig("StepFunctionDGDerivatives.png", dpi=300)
    plt.close()