from copy import deepcopy


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                               TURBINE_LOADER                                 #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

fi_list = {}

layout_x = []
layout_y = []
turbine_cpct = []
turbine_hh = []
turbine_D = []

for fi in fi_list:
    # Wind farm layout
    layout_x.extend(fi.layout_x)
    layout_y.extend(fi.layout_y)

    # Turbine properties
    num_turbs = len(fi.layout_x)
    t0 = fi.floris.farm.turbines[0]
    turbine_cpct.extend([t0.power_thrust_table] * num_turbs)
    turbine_hh.extend([t0.hub_height] * num_turbs)
    turbine_D.extend([t0.rotor_diameter] * num_turbs)

fi = deepcopy(fi_list[0])

# Copy default turbines to new layout locations
fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])

# Update turbine properties
for ti in range(len(fi.layout_x)):
    fi.change_turbine(
        [ti],
        {
            "power_thrust_table": turbine_cpct[ti],
            "rotor_diameter": turbine_D[ti],
            "hub_height": turbine_hh[ti]
        }
    )
