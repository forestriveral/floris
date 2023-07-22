from windrose import WindroseAxes
from matplotlib import pyplot as plt
import windrose
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# Create wind speed and direction variables

# ws = np.random.random(500) * 6
# wd = np.random.random(500) * 360
ws = pd.read_csv(f"parameters/wind_horns_3_1.csv", header=0).values[:, -1]
wd = np.linspace(0, 330, int(360 / 1))

ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
# windrose.wrbar(wd, ws, nsector=360, normed=True, edgecolor='white')
# ax.contourf(wd, ws, bins=np.arange(0, 7, 1), cmap=cm.hot)
ax.set_legend()

plt.show()