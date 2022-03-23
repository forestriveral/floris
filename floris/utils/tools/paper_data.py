import os
import numpy as np
import pandas as pd

from attrs import define, field


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
                os.path.join(file_dir, f"{baseline_data_dir}/WP_2015/Fig_6/LES_OBS_{int(direction)}.csv"), header=0)
            if sectors:
                cols = []
                for s in sectors:
                    cols.extend([f"{direction}+{int(s)}+LES", f"{direction}+{int(s)}+OBS"])
                return data[cols]
            else:
                return data.iloc[:, :8]
        return turbine_array(direction), power_data(direction, sectors)


class AV_2018(object):

    layout = {"Lgd": "Lillgrund",
              "Anh": "Anholt",
              "Nkr": "Nørrekær"}

    direction = {"Lillgrund": [42., 75., 90., 120., 180., 222., 255., 270., 300.],
                 "Anholt": [75., 255.],
                 "Nørrekær": [168., 179., 183., 228., 339., 341.,]}

    turbine = {"Lillgrund": {42.: [16, 17, 18, 19, 20, 21, 22, 23],
                             75.: [3, 11, 20, 28, 36],
                             90.: [8, 25, 38, 48],
                             120.: [2, 9, 17, 25, 32, 37, 42, 46],
                             180.: [15, 22, 28, 39, 43, 36],
                             222.: [23, 22, 21, 20, 19, 18, 17, 16],
                             255.: [36, 28, 20, 11, 3],
                             270.: [48, 38, 25, 8],
                             300.: [46, 42, 37, 32, 25, 17, 9, 2]},
               "Anholt": {75.: [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                          255.: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]},
               "Nørrekær": {168.: [45, 46, 47, 48, 49],
                            179.: [32, 33, 34, 35, 36],
                            183.: [1, 2, 3, 4, 5],
                            228.: [1, 31, 32, 44, 45, 67, 68, 77, 78, 79, 85, 86],
                            339.: [65, 64, 63, 62, 61],
                            341.: [30, 29, 28, 27, 26, 25, 24, 23, 22, 21]}, }

    turbine_sector = {"Lillgrund": {25: [25, 55], 9: [65, 80], 18: [85, 100],
                                    17: [100, 140], 28: [165, 190], 22: [210, 235],
                                    27: [245, 265], 34: [265, 280], 39: [280, 315],
                                    22: [25, 55], 45: [65, 85], 44: [85, 100],
                                    43: [100, 140], 39: [160, 190], 18: [205, 235],
                                    11: [245, 265], 8: [265, 280], 17: [280, 320]},
                      "Anholt": {46: [155, 180], 34: [165, 190], 3: [160, 205], 43: [215, 240],
                                 63: [320, 360], 109: [320, 360], 48: [155, 180], 35: [165, 190],
                                 4: [160, 205], 86: [210, 240], 62: [320, 360], 107: [320, 360]},
                      "Nørrekær": {11: [55, 95], 2: [55, 95], 2: [235, 275]}, 11: [235, 275]}

    def __init__(self,):
        pass

    @classmethod
    def array_power_data(self, layout, direction):
        assert layout in self.layout.keys()
        assert direction in self.turbine[self.layout[layout]].keys()
        turbine_ind = self.turbine[self.layout[layout]][direction]
        data_file = f"{baseline_data_dir}/AV_2018/{self.layout[layout]}/{layout}_wd_{direction}"
        array_power = np.around(np.loadtxt(f"{data_file}.txt"), 4)
        return turbine_ind, array_power

    @classmethod
    def sector_power_data(self, layout, turbine):
        assert layout in self.layout.keys()
        assert turbine in self.turbine_sector[self.layout[layout]].keys()
        direction_range = self.turbine_sector[self.layout[layout]][turbine]
        data_file = f"{baseline_data_dir}/AV_2018/{self.layout[layout]}/{layout}_t_{turbine}"
        turbine_power = np.around(np.loadtxt(f"{data_file}.txt"), 4)
        return direction_range, turbine_power

    @staticmethod
    def combined_data_load():
        # Load the power data with upper and lower limits.
        pass



if __name__ == "__main__":
    # wts, pd = WP_2015.Fig_6(270, [1, 5])
    # print(wts, pd)
    ind = AV_2018.array_power_data("Lgd", 42)
    print(ind)

