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
    # co-ordinate and position arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    pos_x_plot, pos_y_plot, pos_z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    # considering the connectivity to be like a chain!!!
    for i in range(0, n_el):
        x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
        y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
        z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
        pos_x_plot[2*i], pos_x_plot[(2*i)+1] = pos_x[i], pos_x[i+1]
        pos_y_plot[2*i], pos_y_plot[(2*i)+1] = pos_y[i], pos_y[i+1]
        pos_z_plot[2*i], pos_z_plot[(2*i)+1] = pos_z[i], pos_z[i+1]
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"positions": (pos_x_plot, pos_y_plot, pos_z_plot)})

def write_displacements_vtk(output_file, system):
    # co-ordinates and state of the nodes
    nodes = system.weak_form.function_space.nodes
    state = system.state
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    n_nodes = nodes.shape[0]
    n_el = n_nodes-1
    # co-ordinate and displacement arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    disp_x_plot, disp_y_plot, disp_z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    # considering the connectivity to be like a chain!!!
    for i in range(0, n_el):
        x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
        y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
        z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
        disp_x_plot[2*i], disp_x_plot[(2*i)+1] = pos_x[i] - x[i], pos_x[i+1] - x[i+1]
        disp_y_plot[2*i], disp_y_plot[(2*i)+1] = pos_y[i] - y[i], pos_y[i+1] - y[i+1]
        disp_z_plot[2*i], disp_z_plot[(2*i)+1] = pos_z[i] - z[i], pos_z[i+1] - z[i+1]
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"displacements": (disp_x_plot, disp_y_plot, disp_z_plot)})

def write_displacements_forces_vtk(output_file, system):
    # co-ordinates, state and internal forces of the nodes
    nodes = system.weak_form.function_space.nodes
    state = system.state
    internal_forces = system.internal_forces
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    # internal forces of the nodes
    internal_loads_x, internal_loads_y, internal_loads_z = internal_forces[:, 0:1].flatten(), \
                                    internal_forces[:, 1:2].flatten(), internal_forces[:, 2:3].flatten()
    internal_moments_x, internal_moments_y, internal_moments_z = internal_forces[:, 3:4].flatten(), \
                                    internal_forces[:, 4:5].flatten(), internal_forces[:, 5:6].flatten()
    n_nodes = nodes.shape[0]
    n_el = n_nodes-1
    # co-ordinate, displacement and force arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    disp_x_plot, disp_y_plot, disp_z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot = np.zeros(2*n_el), np.zeros(2*n_el), np.zeros(2*n_el)
    # considering the connectivity to be like a chain!!!
    for i in range(0, n_el):
        x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
        y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
        z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
        disp_x_plot[2*i], disp_x_plot[(2*i)+1] = pos_x[i] - x[i], pos_x[i+1] - x[i+1]
        disp_y_plot[2*i], disp_y_plot[(2*i)+1] = pos_y[i] - y[i], pos_y[i+1] - y[i+1]
        disp_z_plot[2*i], disp_z_plot[(2*i)+1] = pos_z[i] - z[i], pos_z[i+1] - z[i+1]
        internal_loads_x_plot[2*i], internal_loads_x_plot[(2*i)+1], internal_moments_x_plot[2*i], internal_moments_x_plot[(2*i)+1] = \
            internal_loads_x[i], internal_loads_x[i+1], internal_moments_x[i], internal_moments_x[i+1]
        internal_loads_y_plot[2*i], internal_loads_y_plot[(2*i)+1], internal_moments_y_plot[2*i], internal_moments_y_plot[(2*i)+1] = \
            internal_loads_y[i], internal_loads_y[i+1], internal_moments_y[i], internal_moments_y[i+1]
        internal_loads_z_plot[2*i], internal_loads_z_plot[(2*i)+1], internal_moments_z_plot[2*i], internal_moments_z_plot[(2*i)+1] = \
            internal_loads_z[i], internal_loads_z[i+1], internal_moments_z[i], internal_moments_z[i+1]
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"displacements": (disp_x_plot, disp_y_plot, disp_z_plot), \
                                                    "forces": (internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot), \
                                                    "moments": (internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot)})