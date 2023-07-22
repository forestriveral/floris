import os, sys
import numpy as np
import matplotlib.pyplot as plt

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
from floris.tools.visualization import (
    calculate_horizontal_plane_with_turbines,
    visualize_cut_plane,
)


fi = FlorisInterface("parameters/gch.yaml")

horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
    yaw_angles=np.array([[[25.,0.,0.]]]),
)

y_plane = fi.calculate_y_plane(
    x_resolution=200,
    z_resolution=100,
    crossstream_dist=630.0,
    yaw_angles=np.array([[[25.,0.,0.]]]),
)
cross_plane = fi.calculate_cross_plane(
    y_resolution=100,
    z_resolution=100,
    downstream_dist=630.0,
    yaw_angles=np.array([[[25.,0.,0.]]]),
)

# Create the plots
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
wakeviz.visualize_cut_plane(horizontal_plane, ax=ax_list[0], title="Horizontal")
wakeviz.visualize_cut_plane(y_plane, ax=ax_list[1], title="Streamwise profile")
wakeviz.visualize_cut_plane(cross_plane, ax=ax_list[2], title="Spanwise profile")

# # Some wake models may not yet have a visualization method included, for these cases can use
# # a slower version which scans a turbine model to produce the horizontal flow
# horizontal_plane_scan_turbine = calculate_horizontal_plane_with_turbines(
#     fi,
#     x_resolution=20,
#     y_resolution=10,
#     yaw_angles=np.array([[[25.,0.,0.]]]),
# )

# fig, ax = plt.subplots()
# visualize_cut_plane(
#     horizontal_plane_scan_turbine,
#     ax=ax,
#     title="Horizontal (coarse turbine scan method)",
# )

# # FLORIS further includes visualization methods for visualing the rotor plane of each
# # Turbine in the simulation

# # Run the wake calculation to get the turbine-turbine interfactions
# # on the turbine grids
# fi.calculate_wake()

# # Plot the values at each rotor
# fig, axes, _ , _ = wakeviz.plot_rotor_values(
#     fi.floris.flow_field.u,
#     wd_index=0,
#     ws_index=0,
#     n_rows=1,
#     n_cols=3,
#     return_fig_objects=True
# )
# fig.suptitle("Rotor Plane Visualization, Original Resolution")

# # Increase the resolution of points on each turbien plane
# solver_settings = {
#     "type": "turbine_grid",
#     "turbine_grid_points": 10
# }
# fi.reinitialize(solver_settings=solver_settings)

# # Run the wake calculation to get the turbine-turbine interfactions
# # on the turbine grids
# fi.calculate_wake()

# # Plot the values at each rotor
# fig, axes, _ , _ = wakeviz.plot_rotor_values(
#     fi.floris.flow_field.u,
#     wd_index=0,
#     ws_index=0,
#     n_rows=1,
#     n_cols=3,
#     return_fig_objects=True
# )
# fig.suptitle("Rotor Plane Visualization, 10x10 Resolution")

wakeviz.show_plots()

