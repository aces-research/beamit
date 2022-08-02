import numpy as np
from pyevtk.hl import linesToVTK

def write_positions_vtk(output_file, system):
    # co-ordinates and state of the nodes
    nodes = system.weak_form.function_space.nodes
    state = system.state
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    n_nodes = nodes.shape[0]
    n_el = n_nodes-1
    x_plot, y_plot, z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    pos_x_plot, pos_y_plot, pos_z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    # co-ordinate and position arrays for writing the VTK file
    # considering the connectivity to be like a chain!!!
    for i in range(0, n_el):
        x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
        y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
        z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
        pos_x_plot[2*i], pos_x_plot[(2*i)+1] = pos_x[i], pos_x[i+1]
        pos_y_plot[2*i], pos_y_plot[(2*i)+1] = pos_y[i], pos_y[i+1]
        pos_z_plot[2*i], pos_z_plot[(2*i)+1] = pos_z[i], pos_z[i+1]
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"positions": (pos_x_plot, pos_y_plot, pos_z_plot)})