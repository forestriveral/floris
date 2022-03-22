import os
import matplotlib.pyplot as plt
import numpy as np



class WindFarmLayout(object):

    def __init__(self, ):
        pass

    @classmethod
    def reader(self, txt_file):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        layout = np.around(np.loadtxt(f"../inputs/layouts/{txt_file}.txt"), 2)
        return (list(layout[:, 0]), list(layout[:, 1]))

    @classmethod
    def plot(self, layout, annotate=None):
        x, y = np.array(layout[0]), np.array(layout[1])
        xb, yb = (np.max(x) - np.min(x)) / 5, (np.max(y) - np.min(y)) / 5
        plt.figure(dpi=100)
        plt.scatter(x, y)
        plt.xlim([np.min(x) - xb, np.max(x) + xb])
        plt.ylim([np.min(y) - yb, np.max(y) + yb])
        if annotate is not None:
            num_labs = [str(i) for i in range(1, 81)]
            for i in range(len(num_labs)):
                plt.annotate(num_labs[i], xy=(x[i], y[i]),
                            xytext=(x[i] + 50, y[i] + 50))
        # plt.xlim((-8 * 80., 80 * 80.));plt.ylim((-4 * 80., 70 * 80.))
        plt.show()

    @classmethod
    def layout(self, farm):
        wf_name = {"HR1": "Horns Rev 1",
                   "HR2": "Horns Rev 2",
                   "Anh": "Anholt",
                   "Lgd": "Lillgrund",
                   "Nys": "Nysted",
                   "LoA": "London Array",
                   "Rdd": "Rodsand II",
                   "NrH": "North Hoyle",
                   "Nkr": "Nørrekær",
                   "template": "template",}
        return self.reader(wf_name[farm])


def HornsRev1_generator(): # HR1
        c_n, r_n = 8, 10
        labels = []
        for i in range(1, r_n + 1):
            for j in range(1, c_n + 1):
                l = "c{}_r{}".format(j, i)
                labels.append(l)
        # Wind turbines location generating  wt_c1_r1 = (0., 4500.)
        layout = np.zeros((c_n * r_n, 2))
        num = 0
        for i in range(r_n):
            for j in range(c_n):
                layout[num, :] = [0. + 68.589 * j + 7 * 80. * i, 3911. - j * 558.616]
                num += 1
        # np.savetxt('./layouts/Horns Rev 1.txt', np.around(layout, 2))
        return (list(np.around(layout, 2)[:, 0]), list(np.around(layout, 2)[:, 1]))



if __name__ == "__main__":
    
    layout = WindFarmLayout.layout('HR2')
    print(layout)
    # WindFarmLayout.plot(layout)