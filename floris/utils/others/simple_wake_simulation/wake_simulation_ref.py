import numpy as np
import matplotlib.pyplot as plt

from floris.tools import FlorisInterface
import floris.tools.visualization as wakeviz

fi = FlorisInterface("../../input/config/jensen.yaml")
fi.reinitialize(wind_directions=[270.], wind_speeds=[8.0], turbulence_intensity=0.07,
                layout_x=[0, 500.], layout_y=[0., 0.])
yaw_angles = np.array([[[10., 15.]]])
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers() / 1000.
print('turbine_powers: ', turbine_powers[0, 0])

# horizontal_plane = fi.calculate_horizontal_plane(
#     x_resolution=200,
#     y_resolution=100,
#     height=90.0,
#     yaw_angles=np.array([[[10.,15.]]]),
# )

# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# wakeviz.visualize_cut_plane(horizontal_plane, ax=ax, title="Horizontal")
# wakeviz.plot_turbines_with_fi(fi, ax=ax,)
# plt.show()