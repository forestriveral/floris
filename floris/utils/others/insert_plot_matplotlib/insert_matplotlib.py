import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

x = np.linspace(-0.1*np.pi, 2*np.pi, 30)
y_1 = np.sinc(x)+0.7
y_2 = np.tanh(x)
y_3 = np.exp(-np.sinc(x))

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
ax.plot(x, y_1, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C0')

ax.plot(x, y_2, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C3')

ax.plot(x, y_3, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C2')

ax.legend(labels=["y_1", "y_2","y_3"], ncol=3)

# axins = ax.inset_axes((0.2, 0.2, 0.4, 0.3))
axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.5, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)

axins.plot(x, y_1, color='k', linestyle=':', linewidth=1,
            marker='o', markersize=5,
            markeredgecolor='black', markerfacecolor='C0')

axins.plot(x, y_2, color='k', linestyle=':', linewidth=1,
            marker='o', markersize=5,
            markeredgecolor='black', markerfacecolor='C3')

axins.plot(x, y_3, color='k', linestyle=':', linewidth=1,
            marker='o', markersize=5,
            markeredgecolor='black', markerfacecolor='C2')

# 设置放大区间
zone_left = 11
zone_right = 12

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5 # x轴显示范围的扩展比例
y_ratio = 0.5 # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((y_1[zone_left:zone_right], y_2[zone_left:zone_right], y_3[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

plt.show()