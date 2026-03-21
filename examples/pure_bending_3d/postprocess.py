import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# elastic modulus of beam
E = 2.0E11
# length of beam
L = 1.0
# beam slenderness ratio
slenderness_ratio = 10.0
# side of the square cross-section
a = L / slenderness_ratio
# moment of inertia
I = (a**4) / 12.0
# applied moment
TIP_MOMENT = 1.0E07
# helix radius
helix_radius = (E * I) / (2.0 * TIP_MOMENT)

def compute_analytical_positions(arc_length_coordinates):
    beta = arc_length_coordinates / (np.sqrt(2.0) * helix_radius)
    x = (helix_radius / np.sqrt(2.0)) * (beta + np.sin(beta))
    y = helix_radius * (1.0 - np.cos(beta))
    z = (helix_radius / np.sqrt(2.0)) * (beta - np.sin(beta))
    return np.column_stack((x, y, z))

def compute_position_error_norms(output_dir):
    # quad points to integrate the error norm
    quad_points, quad_weights = np.polynomial.legendre.leggauss(2)
    # linear shape functions at the quad points
    shape_functions = np.array([
        0.5 * (1.0 - quad_points),
        0.5 * (1.0 + quad_points)
    ]).T
    # loop over the folders in the output directory
    element_lengths = []
    error_norms = []
    for folder in sorted(os.listdir(output_dir)):
        folder_path = os.path.join(output_dir, folder)
        # loop over the files in the folder
        files_list = os.listdir(folder_path)
        files_list.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
        final_file_path = os.path.join(folder_path, files_list[-1])
        # read the mesh
        mesh = pv.read(final_file_path)
        # extract the nodal coordinates
        nodal_coordinates = mesh.points
        # get the element length
        element_length = nodal_coordinates[1, 0] - nodal_coordinates[0, 0]
        # compute the jacobian
        jacobian = element_length / 2.0
        # get the number of elements
        num_elements = int(nodal_coordinates.shape[0] / 2)
        # nodal displacements
        nodal_displacements = mesh["displacements"]
        error_norm = 0.0
        # loop over the elements
        for i in range(num_elements):
            # get the arc length coordinate of the first node
            node1_arc_coord = nodal_coordinates[2*i, 0]
            # get the arc-length coordinates of the quad points
            quad_arc_coords = node1_arc_coord + 0.50 * element_length * (1.0 + quad_points)
            # compute the analytical positions
            analytical_positions = compute_analytical_positions(
                quad_arc_coords)
            # compute the nodal positions at the quad points
            numerical_positions = shape_functions @ (
                nodal_coordinates[2*i:2*i+2, :] + nodal_displacements[2*i:2*i+2, :])
            # compute the error norm for this element
            error_norm_element = np.linalg.norm(numerical_positions - analytical_positions,
                                                axis=1, keepdims=False) * jacobian * quad_weights
            error_norm_element = np.sum(error_norm_element)
            error_norm += error_norm_element
        element_lengths.append(element_length)
        error_norms.append(error_norm)
    # sort the results
    sorted_indices = np.argsort(element_lengths)
    element_lengths = np.array(element_lengths)[sorted_indices]
    error_norms = np.array(error_norms)[sorted_indices]
    return element_lengths, error_norms

if __name__ == "__main__":
    
    # compute the position error norms
    element_lengths_CG, error_norms_CG = compute_position_error_norms(
        "./VTK-actual/VTK-CG")
    element_lengths_DG, error_norms_DG = compute_position_error_norms(
        "./VTK-actual/VTK-DG")

    # plot the results
    fig = plt.figure(figsize=(10.98, 9.0))
    plt.rc("font", size=28)
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.loglog(element_lengths_CG, error_norms_CG, linewidth=6.0, markersize=15,
            color='green', marker='o', label=r"\bf{CG}")
    plt.loglog(element_lengths_DG, error_norms_DG, linewidth=3.0, markersize=15,
               color='blue', marker='x', label=r"\bf{DG}")
    plt.loglog(element_lengths_CG, (element_lengths_CG**2.0), color='black', linestyle='--',
            linewidth=3.0, label=r"$2^{nd}\:\:order$")
    plt.xlabel(r"\bf{Mesh size} $(h)$")
    plt.ylabel(r"\bf{Relative error} $(||e||^{\:2}_{\:rel})$")
    plt.grid(True, which='both', linestyle='-', linewidth=1.0)
    plt.legend()
    fig.savefig("ConvergencePlots.png", dpi=300)
