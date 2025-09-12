import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# length of beam (45 degree bend of radius 100)
L = (100.0*np.pi)/4
# applied load
TIP_LOAD = 2000.0

# output directory
output_dir = "./VTK"
output_files = glob.glob(f"{output_dir}/output-*.vtu")
output_files.sort(key=lambda x: int(x.split("output-")[-1].split(".")[0]))
output_files = output_files[1:]  # skip the initial state

# read the displacements of the initial state
initial_state = pv.read(output_files[0])
nodal_coordinates = initial_state.points
beam_tip_index = np.where(nodal_coordinates[:, 0] == L)[0][0]
initial_tip_displacements = initial_state["displacements"][beam_tip_index, :]

# loop over the output files and compute the displacements
tip_displacement_x = []
tip_displacement_y = []
tip_displacement_z = []
applied_load = np.linspace(0.0, TIP_LOAD, len(output_files))
for i in range(len(output_files)):
    # read the output file
    output = pv.read(output_files[i])
    # extract the displacements
    tip_displacements = output["displacements"][beam_tip_index, :] - initial_tip_displacements
    # append the displacements
    tip_displacement_x.append(tip_displacements[0])
    tip_displacement_y.append(tip_displacements[1])
    tip_displacement_z.append(tip_displacements[2])
tip_displacement_x = np.array(tip_displacement_x)
tip_displacement_y = np.array(tip_displacement_y)
tip_displacement_z = np.array(tip_displacement_z)

# open the reference solution file
reference_load_displacement_x = np.loadtxt("./simo_et_al_1986/u1.txt")
reference_load_displacement_y = np.loadtxt("./simo_et_al_1986/u2.txt")
reference_load_displacement_z = np.loadtxt("./simo_et_al_1986/u3.txt")

# plot the results
fig = plt.figure(figsize=(10.98, 9.0))
plt.rc("font", size=20)
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.scatter(reference_load_displacement_x[:, 0], reference_load_displacement_x[:, 1],
            label=r"\bf{u}$_{1}$: Simo et al.", color='black', marker='^', s=100)
plt.scatter(reference_load_displacement_y[:, 0], reference_load_displacement_y[:, 1],
            label=r"\bf{u}$_{2}$: Simo et al.", color='blue', marker='x', s=100)
plt.scatter(reference_load_displacement_z[:, 0], reference_load_displacement_z[:, 1],
            label=r"\bf{u}$_{3}$: Simo et al.", color='green', marker='s', s=100)
plt.plot(applied_load, tip_displacement_x,
         label=r"\bf{u}$_{1}$: Present", color='black', linewidth=3.0)
plt.plot(applied_load, tip_displacement_y,
         label=r"\bf{u}$_{2}$: Present", linestyle='-.', color='blue', linewidth=3.0)
plt.plot(applied_load, tip_displacement_z,
         label=r"\bf{u}$_{3}$: Present", linestyle='--', color='green', linewidth=3.0)
plt.xlabel(r"\bf{Load} $(N)$")
plt.ylabel(r"\bf{Tip displacement} $(m)$")
ymax = max(abs(plt.ylim()[0]), abs(plt.ylim()[1]))
plt.ylim(-ymax, ymax)
plt.grid()
plt.legend(ncol=2, loc='center right')
fig.savefig("LoadVsDisplacement.png", dpi=300)
