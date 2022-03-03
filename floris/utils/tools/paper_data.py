import os
import numpy as np
import pandas as pd


baseline_data_dir = "../data/baselines"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                    2015_WP                                   #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class WP_2015(object):

    def __init__(self,):
        pass

    @classmethod
    def Fig_6(self, direction, sectors):
        def turbine_array(direction):
            assert direction in [270, 222, 312], \
                "Invalid wind direction in WP_2015!"
            if direction == 270:
                wt_array_270 = np.array([1, 9, 17, 25, 33, 41, 49, 57])
                wts_270 = np.zeros((8, 8))
                for i in range(8):
                    wts_270[i, :] = wt_array_270 + i * 1
                return wts_270.astype(np.int)
            if direction == 222:
                wt_array_222 = np.array([8, 15, 22, 29, 36])
                wts_222 = np.zeros((8, 5))
                wts_222[3, :] = wt_array_222
                for i in range(3):
                    wts_222[i, :] = wt_array_222 - (3 - i)
                for i in range(4):
                    wts_222[4 + i] = wt_array_222 + 8 * (i + 1)
                return wts_222.astype(np.int)
            else:
                wt_array_312 = np.array([1, 10, 19, 28, 37])
                wts_312 = np.zeros((8, 5))
                wts_312[4, :] = wt_array_312
                for i in range(4):
                    wts_312[i, :] = wt_array_312 + 8 * (4 - i)
                for i in range(3):
                    wts_312[5 + i] = wt_array_312 + (i + 1)
                return wts_312.astype(np.int)

        def power_data(direction, sectors):
            assert len(set(sectors + [1., 5., 10., 15.])) == 4, \
                "Invalid wind sector in WP_2015!"
            file_dir = os.path.dirname(os.path.abspath(__file__))
            data = pd.read_csv(
                os.path.join(file_dir, f"../data/baselines/WP_2015/Fig_6/LES_OBS_{int(direction)}.csv"), header=0)
            if sectors:
                cols = []
                for s in sectors:
                    cols.extend([f"{direction}+{int(s)}+LES", f"{direction}+{int(s)}+OBS"])
                return data[cols]
            else:
                return data.iloc[:, :8]
        return turbine_array(direction), power_data(direction, sectors)




if __name__ == "__main__":
    wts, pd = WP_2015.Fig_6(270, [1, 5])
    print(wts, pd)

