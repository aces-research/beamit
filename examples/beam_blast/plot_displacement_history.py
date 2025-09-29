import sys
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# simulation parameters
time_step = 1.0E-08
beam_length = 0.2032

# get the location of the output files from the command line arguments
if (len(sys.argv) < 2):
    raise ValueError(
        "Provide the location of the output files as a command line argument.")
output_dir = sys.argv[1]
output_files_list = os.listdir(output_dir)
output_files_list.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))

# generate the time array
time_array = []
for i in range(len(output_files_list)):
    # extract the time step number from the file name
    time_step_number = int(output_files_list[i].split('-')[1].split('.')[0])
    time_array.append(time_step_number * time_step)
time_array = np.array(time_array)

displacement_array = []
for output_file in output_files_list:
    # read the mesh
    mesh = pv.read(os.path.join(output_dir, output_file))
    # extract the nodal coordinates
    nodal_coordinates = mesh.points
    # get the displacement field
    displacements = mesh["displacements"]
    # extract the displacements of the mid-span node
    mid_span_displacements = displacements[np.abs(
        nodal_coordinates[:, 0] - 0.50 * beam_length) < 1.0E-10]
    displacement_array.append(np.mean(mid_span_displacements[:, 2]))
displacement_array = np.array(displacement_array)

# load the reference data
reference_shell_data = np.loadtxt("shell_reference_data/displacement_history_shell_400Pas.txt")

# plot the displacement history
fig = plt.figure(figsize=(10.98, 9.0))
plt.rc("font", size=20)
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.plot(time_array * 1.0E3, displacement_array * 1.0E3, color='blue', linestyle='-', 
         linewidth=3.0, label=r"\bf{Present}")
plt.scatter(reference_shell_data[0::100, 0] * 1.0E3, reference_shell_data[0::100, 1] * 1.0E3, 
            color='red', marker='o', s=25, label=r"\bf{Talamini et al. (\emph{Shell})}")
plt.xlabel(r"\bf{Time (ms)}")
plt.ylabel(r"\bf{Mid span displacement Z (mm)}")
plt.grid()
plt.legend(loc='upper left', fontsize=18)
fig.savefig("DisplacementHistory.png", dpi=300)
