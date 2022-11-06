import numpy as np
import pandas as pd
import itertools

from floris.utils.visual import layout_opt_plot_old as layout_plot

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                               WIND_FARMS                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Horns(object):

    wind_pdf = np.array([[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
                            [8.89, 9.27, 8.23, 9.78, 11.64, 11.03, 11.50,
                                11.92, 11.49, 11.08, 11.34, 10.76, 8.89],
                            [2.09, 2.13, 2.29, 2.30, 2.67, 2.45,
                                2.51, 2.40, 2.35, 2.27, 2.24, 2.19, 2.09],
                            [4.82, 4.06, 3.59, 5.27, 9.12, 6.97, 9.17,
                            11.84, 12.41, 11.34, 11.70, 9.69, 4.82]])

    ref_pdf = {'single': np.array([[1.90, 1.90, 1.90, 1.90, 1.90, 1.90, 1.90,
                                    1.90, 79.10, 1.90, 1.90, 1.90, 1.90]]),
               'average': np.array([[8.33, 8.33, 8.33, 8.33, 8.33, 8.33, 8.33,
                                     8.33, 8.33, 8.33, 8.33, 8.33, 8.33]])}

    @classmethod
    def layout(cls):
        # Wind turbines labelling
        c_n, r_n = 8, 10
        labels = []
        for i, j in itertools.product(range(1, r_n + 1), range(1, c_n + 1)):
            l = "c{}_r{}".format(j, i)
            labels.append(l)
        # Wind turbines location generating  wt_c1_r1 = (0., 4500.)
        locations = np.zeros((c_n * r_n, 2))
        num = 0
        for num, (i, j) in enumerate(itertools.product(range(r_n), range(c_n))):
            loc_x = 0. + 68.589 * j + 7 * 80. * i
            loc_y = 3911. - j * 558.616
            locations[num, :] = [loc_x, loc_y]
        return np.array(locations)

    @classmethod
    def params(cls):
        params = {}
        params = {"D_r": [80.0],
                  "z_hub": [70.0],
                  "v_in": [4.0],
                  "v_rated": [15.0],
                  "v_out": [25.0],
                  "P_rated": [2.0],
                  "power_curve": ["horns"],
                  "ct_curve": ["horns"]}
        return pd.DataFrame(params)

    @classmethod
    def pow_curve(cls, vel):
        if vel <= 4.:
            return 0.
        elif vel >= 15.:
            return 2.
        else:
            return 1.45096246e-07 * vel**8 - 1.34886923e-05 * vel**7 + \
                5.23407966e-04 * vel**6 - 1.09843946e-02 * vel**5 + \
                    1.35266234e-01 * vel**4 - 9.95826651e-01 * vel**3 + \
                        4.29176920e+00 * vel**2 - 9.84035534e+00 * vel + \
                            9.14526132e+00

    @classmethod
    def ct_curve(cls, vel):
        if vel <= 10.:
            vel = 10.
        elif vel >= 20.:
            vel = 20.
        return np.array([-2.98723724e-11, 5.03056185e-09, -3.78603307e-07,  1.68050026e-05,
                            -4.88921388e-04,  9.80076811e-03, -1.38497930e-01,  1.38736280e+00,
                            -9.76054549e+00,  4.69713775e+01, -1.46641177e+02,  2.66548591e+02,
                            -2.12536408e+02]).dot(np.array([vel**12, vel**11, vel**10, vel**9,
                                                            vel**8, vel**7, vel**6, vel**5,
                                                            vel**4, vel**3, vel**2, vel, 1.]))

    @classmethod
    def baseline(cls, Nt, bounds=None, arrays=None, grids=None):
        return baseline_layout(Nt, bounds=bounds, arrays=arrays, grids=grids)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def grid_points(glens, lens, num=20):
    points = np.zeros((num, lens))
    points[0, :] = np.around(np.linspace(0, glens - 1, lens))
    points[1, :] = np.ceil(np.linspace(0, glens - 1, lens))
    points[2, :] = np.floor(np.linspace(0, glens - 1, lens))
    for i in range(3, num):
        points[i, :] = np.sort(np.random.choice(np.arange(glens), lens, replace=False))
    return points


def grid_baseline(grids, wtnum, seed=None):
    default_arrays = {9:(3, 3), 16:(4, 4), 25:(5, 5), 30:(5, 6), 36:(6, 6),
                      42:(6, 7), 49:(7, 7), 56:(7, 8), 64:(8, 8), 72:(8, 9),
                      80:(8, 10)}
    if wtnum not in default_arrays.keys():
        return None
    yl, xl = default_arrays[wtnum]
    np.random.seed(seed)
    xs, ys = grid_points(grids.shape[2], xl).astype(np.int), \
        grid_points(grids.shape[1], yl).astype(np.int)
    assert xs.shape[0] == ys.shape[0]
    baselines = np.zeros((xs.shape[0], wtnum))
    for i in range(xs.shape[0]):
        grid_mask = np.full((grids.shape[1], grids.shape[2]), False)
        xx, yy = np.meshgrid(xs[i, :], ys[i, :][::-1])
        grid_mask[yy, xx] = True
        baselines[i, :] = np.arange(grids.shape[1] * grids.shape[2])[grid_mask.ravel()]
    return baselines


def baseline_layout(Nt, bounds=None, arrays=None, grids=None, **kwargs):
    if grids is not None:
        return grid_baseline(grids, Nt)
    if not arrays:
        default_arrays = {9:(3, 3), 16:(4, 4), 25:(5, 5), 30:(5, 6),
                          36:(6, 6), 42:(6, 7), 49:(7, 7), 56:(7, 8),
                          64:(8, 8), 72:(8, 9), 80:(8, 10)}
        if (Nt is None) or (Nt not in default_arrays.keys()):
            print(f"No default baseline layout of {Nt}")
            return None
        wt_array = default_arrays[Nt]
    else:
        wt_array = arrays
    Nh, Nv = wt_array[0], wt_array[1]
    assert Nt == Nh * Nv, 'Nt should be equal to Nh * Nv'
    bounds = bounds if bounds is not None else \
        np.array([[0, 5040, 5040, 0], [3911, 3911, 0, 0]])
    xs = np.linspace(bounds[0, 0], bounds[0, 1], Nh)
    ys = np.linspace(bounds[1, 2], bounds[1, 1], Nv)
    xs, ys = np.meshgrid(xs, ys[::-1])
    xy = np.concatenate((xs.ravel()[None, :], ys.ravel()[None, :])).transpose(1, 0)
    return (xy, [xs[0, 1] - xs[0, 0], ys[0, 0] - ys[1, 0]]) \
        if kwargs.get('spacing', False) else xy


if __name__ == "__main__":
    layout = Horns.baseline(49)
    print(layout.shape)
    layout_plot(baseline=layout, layout_name='baseline_49')