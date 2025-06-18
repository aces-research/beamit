import numpy as np
from pyevtk.hl import linesToVTK
from beamit import Material
from beamit import WeakForm

def write_positions_vtk(output_file, system):
    # co-ordinates, state of the nodes and the discretization type
    nodes = system.weak_form.function_space.nodes
    state = system.state
    discretization_type = system.weak_form.function_space.discretization_type
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    n_el = system.weak_form.function_space.E
    npel = system.weak_form.function_space.npel
    # co-ordinate and position arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    pos_x_plot, pos_y_plot, pos_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    if (discretization_type == "CG"):
        # considering the connectivity to be like a chain!!!
        for i in range(0, n_el):
            x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
            y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
            z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
            pos_x_plot[2*i], pos_x_plot[(2*i)+1] = pos_x[i], pos_x[i+1]
            pos_y_plot[2*i], pos_y_plot[(2*i)+1] = pos_y[i], pos_y[i+1]
            pos_z_plot[2*i], pos_z_plot[(2*i)+1] = pos_z[i], pos_z[i+1]
    elif (discretization_type == "DG"):
        x_plot, y_plot, z_plot = x, y, z
        pos_x_plot, pos_y_plot, pos_z_plot = pos_x, pos_y, pos_z
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"positions": (pos_x_plot, pos_y_plot, pos_z_plot)})
    print("\nOutput file is generated.")

def write_displacements_vtk(output_file, system):
    # co-ordinates, state of the nodes and the discretization type
    nodes = system.weak_form.function_space.nodes
    state = system.state
    discretization_type = system.weak_form.function_space.discretization_type
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    n_el = system.weak_form.function_space.E
    npel = system.weak_form.function_space.npel
    # co-ordinate and displacement arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    disp_x_plot, disp_y_plot, disp_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    if (discretization_type == "CG"):
        # considering the connectivity to be like a chain!!!
        for i in range(0, n_el):
            x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
            y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
            z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
            disp_x_plot[2*i], disp_x_plot[(2*i)+1] = pos_x[i] - x[i], pos_x[i+1] - x[i+1]
            disp_y_plot[2*i], disp_y_plot[(2*i)+1] = pos_y[i] - y[i], pos_y[i+1] - y[i+1]
            disp_z_plot[2*i], disp_z_plot[(2*i)+1] = pos_z[i] - z[i], pos_z[i+1] - z[i+1]
    elif (discretization_type == "DG"):
        x_plot, y_plot, z_plot = x, y, z
        disp_x_plot, disp_y_plot, disp_z_plot = pos_x - x, pos_y - y, pos_z - z
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"displacements": (disp_x_plot, disp_y_plot, disp_z_plot)})
    print("\nOutput file is generated.")

def write_displacements_forces_vtk(output_file, system):
    # co-ordinates, state of the nodes and the discretization type
    nodes = system.weak_form.function_space.nodes
    state = system.state
    internal_forces = system.internal_forces
    discretization_type = system.weak_form.function_space.discretization_type
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    # internal forces of the nodes
    internal_loads_x, internal_loads_y, internal_loads_z = internal_forces[:, 0:1].flatten(), \
                                    internal_forces[:, 1:2].flatten(), internal_forces[:, 2:3].flatten()
    internal_moments_x, internal_moments_y, internal_moments_z = internal_forces[:, 3:4].flatten(), \
                                    internal_forces[:, 4:5].flatten(), internal_forces[:, 5:6].flatten()
    n_el = system.weak_form.function_space.E
    npel = system.weak_form.function_space.npel
    # co-ordinate, displacement and force arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    disp_x_plot, disp_y_plot, disp_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    if (discretization_type == "CG"):
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
    elif (discretization_type == "DG"):
        x_plot, y_plot, z_plot = x, y, z
        disp_x_plot, disp_y_plot, disp_z_plot = pos_x - x, pos_y - y, pos_z - z
        internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot = internal_loads_x, internal_loads_y, internal_loads_z
        internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot = internal_moments_x, internal_moments_y, internal_moments_z
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"displacements": (disp_x_plot, disp_y_plot, disp_z_plot), \
                                                    "forces": (internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot), \
                                                    "moments": (internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot)})
    print("\nOutput file is generated.")

def write_output_vtk(output_file, system):
    # co-ordinates, state of the nodes and the discretization type
    nodes = system.weak_form.function_space.nodes
    state = system.state
    internal_forces = system.internal_forces
    discretization_type = system.weak_form.function_space.discretization_type
    # convert the fields to 3D state for post-processing
    if (isinstance(system.weak_form, (WeakForm.EulerBernoulliWeakFormCG))):
        # add zeros to the Y and Z coordinates
        nodes = np.append(nodes, np.zeros([nodes.shape[0], 2]), axis=1)
        # add displacements at the appropriate location
        displacements_x = state[:, 0:1]
        displacements_y = state[:, 1:2]
        state = np.zeros([nodes.shape[0], 6])
        state[:, 0:1] = displacements_x
        state[:, 1:2] = displacements_y
        # add internal forces at the appropriate location
        axial_forces = internal_forces[:, 0:1]
        shear_forces = internal_forces[:, 1:2]
        bending_moments = internal_forces[:, 2:3]
        internal_forces = np.zeros([nodes.shape[0], 6])
        internal_forces[:, 0:1] = axial_forces
        internal_forces[:, 1:2] = shear_forces
        internal_forces[:, 5:6] = bending_moments
    x, y, z = nodes[:, 0:1].flatten(), nodes[:, 1:2].flatten(), nodes[:, 2:3].flatten()
    # positions of the nodes
    pos_x, pos_y, pos_z = state[:, 0:1].flatten(), state[:, 1:2].flatten(), state[:, 2:3].flatten()
    # internal forces of the nodes
    internal_loads_x, internal_loads_y, internal_loads_z = internal_forces[:, 0:1].flatten(), \
                                    internal_forces[:, 1:2].flatten(), internal_forces[:, 2:3].flatten()
    internal_moments_x, internal_moments_y, internal_moments_z = internal_forces[:, 3:4].flatten(), \
                                    internal_forces[:, 4:5].flatten(), internal_forces[:, 5:6].flatten()
    n_el = system.weak_form.function_space.E
    npel = system.weak_form.function_space.npel
    # co-ordinate, displacement, force, and interface damage status arrays for writing to the VTK file
    x_plot, y_plot, z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    disp_x_plot, disp_y_plot, disp_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot = np.zeros(npel*n_el), np.zeros(npel*n_el), np.zeros(npel*n_el)
    damage_status = np.zeros(npel*n_el)
    if (discretization_type == "CG"):
        # considering the connectivity to be like a chain!!!
        for i in range(0, n_el):
            x_plot[2*i], x_plot[(2*i)+1] = x[i], x[i+1]
            y_plot[2*i], y_plot[(2*i)+1] = y[i], y[i+1]
            z_plot[2*i], z_plot[(2*i)+1] = z[i], z[i+1]
            if ((type(system.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormCG) or
                    (type(system.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormDG)):
                disp_x_plot[2*i], disp_x_plot[(2*i)+1] = pos_x[i] - x[i], pos_x[i+1] - x[i+1]
                disp_y_plot[2*i], disp_y_plot[(2*i)+1] = pos_y[i] - y[i], pos_y[i+1] - y[i+1]
                disp_z_plot[2*i], disp_z_plot[(2*i)+1] = pos_z[i] - z[i], pos_z[i+1] - z[i+1]
            else:
                disp_x_plot[2*i], disp_x_plot[(2*i)+1] = pos_x[i], pos_x[i+1]
                disp_y_plot[2*i], disp_y_plot[(2*i)+1] = pos_y[i], pos_y[i+1]
                disp_z_plot[2*i], disp_z_plot[(2*i)+1] = pos_z[i], pos_z[i+1]
            internal_loads_x_plot[2*i], internal_loads_x_plot[(2*i)+1], internal_moments_x_plot[2*i], internal_moments_x_plot[(2*i)+1] = \
                internal_loads_x[i], internal_loads_x[i+1], internal_moments_x[i], internal_moments_x[i+1]
            internal_loads_y_plot[2*i], internal_loads_y_plot[(2*i)+1], internal_moments_y_plot[2*i], internal_moments_y_plot[(2*i)+1] = \
                internal_loads_y[i], internal_loads_y[i+1], internal_moments_y[i], internal_moments_y[i+1]
            internal_loads_z_plot[2*i], internal_loads_z_plot[(2*i)+1], internal_moments_z_plot[2*i], internal_moments_z_plot[(2*i)+1] = \
                internal_loads_z[i], internal_loads_z[i+1], internal_moments_z[i], internal_moments_z[i+1]
    elif (discretization_type == "DG"):
        x_plot, y_plot, z_plot = x, y, z
        if ((type(system.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormCG) or
                (type(system.weak_form) == WeakForm.TFKLGeometricallyExactWeakFormDG)):
            disp_x_plot, disp_y_plot, disp_z_plot = pos_x - x, pos_y - y, pos_z - z
        else:
            disp_x_plot, disp_y_plot, disp_z_plot = pos_x, pos_y, pos_z
        internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot = internal_loads_x, internal_loads_y, internal_loads_z
        internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot = internal_moments_x, internal_moments_y, internal_moments_z
        # write damage status output for a cohesive interface material
        if (isinstance(system.weak_form.material, (Material.TFKLCohesiveInterfaceMaterial))):
            damage_values = system.weak_form.internal_variables[:,2:3]/system.weak_form.material.delta_c
            damage_status[1::2][:-1] = damage_values.flatten()
            damage_status[2::2] = damage_values.flatten()
            # replace the damage status at fully cracked interfaces with 1.0
            damage_status[damage_status > 1.0] = 1.0
    linesToVTK(output_file, x_plot, y_plot, z_plot, pointData = {"displacements": (disp_x_plot, disp_y_plot, disp_z_plot), \
                                                    "forces": (internal_loads_x_plot, internal_loads_y_plot, internal_loads_z_plot), \
                                                    "moments": (internal_moments_x_plot, internal_moments_y_plot, internal_moments_z_plot), \
                                                    "damage": damage_status})
    print("\nOutput file is generated.")