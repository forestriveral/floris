import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties


data = np.random.uniform(0, 1, 80).reshape(20, 4)
final_data = [['%.3f' % j for j in i] for i in data]

# mpl.style.use('seaborn')
# mpl.rc('xtick', labelsize = 7)
# mpl.rc('ytick', labelsize = 7)

fig = plt.figure()

fig.subplots_adjust(left=0.1, wspace=0.5)
plt.subplot2grid((1, 4), (0, 0), colspan=3)

table_subplot = plt.subplot2grid((1, 4), (0, 3))

table = plt.table(cellText=final_data,
                  colLabels=['A', 'B', 'C', 'D'],
                  loc='center', cellLoc='center',
                  colColours=['#FFFFFF', '#F3CC32', '#2769BD', '#DC3735'])
table.auto_set_font_size(False)
table.set_fontsize(7)
table.auto_set_column_width((-1, 0, 1, 2, 3))

for (row, col), cell in table.get_celld().items():
    if (row == 0):
        cell.set_text_props(fontproperties=FontProperties(weight='bold', size=7))
plt.axis('off')

plt.savefig("miscellaneous/test.png", format='png', dpi=300,)
plt.show()
