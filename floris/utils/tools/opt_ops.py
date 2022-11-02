import os
import json
import shutil
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import \
    euclidean_distances

from floris.utils.tools import farm_config as fconfig
from floris.utils.visual.wind_resource import winds_pdf


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  OPTIMIZATION                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def wind_speed_dist(type="weibull"):

    def weibull_pdf(v, scale, shape):
        return (shape / scale) * (v / scale)**(shape - 1) * np.exp(-(v / scale) ** shape)

    def weibull_cdf(v, scale, shape):
        return 1 - np.exp(-(v / scale) ** shape)

    return weibull_pdf, weibull_cdf


def params_loader(name="horns"):
    if name == "horns":
        return fconfig.Horns
    else:
        raise ValueError("Invalid wind farm name! Please check!...")


def winds_loader(data, name, bins, speed=(4, 25)):
    data_path = f"../params/wflo/{data}_{name}_{bins[0]}_{bins[1]}.csv"
    if not os.path.exists(data_path):
        winds_pdf(bins, speed, name=name, output=data)
    return pd.read_csv(data_path, header=0, index_col=0)


def winds_discretization(bins, speeds=(4, 25)):
    vbins, wbins, v_in, v_out = bins[0], bins[1], speeds[0], speeds[1]
    v_bin = np.append(np.arange(v_in, v_out, vbins), v_out)
    v_point = (v_bin[:-1] + v_bin[1:]) / 2
    w_point = - 0.5 * wbins + (np.arange(1, int(360 / wbins) + 1) - 0.5) * wbins
    w_bin = np.append((w_point - 0.5 * wbins), (w_point[-1] + 0.5 * wbins))
    return v_bin, v_point, w_bin, w_point


def deficits_interpolation(deficits, nums=24):
    inter_range = deficits.shape[2] // nums
    assert inter_range >= 1, "Invalid inter range!"
    inter_deficits = np.zeros((deficits.shape[0], deficits.shape[1], nums))
    for i in range(nums):
        inter_deficits[:, :, i] = np.mean(
            deficits[:, :, i * inter_range : (i + 1) * inter_range], axis=2)
    return inter_deficits


def results_packing(fname, subpath=None, path="output", default="solution"):
    today = str(datetime.datetime.now()).split(" ")[0].split("-")
    date = "_".join([today[0][2:], today[1].strip("0"), today[2]])
    dir_name = f"{path}/{date}" if subpath is None else f"{path}/{subpath}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    shutil.copytree(default, f"{dir_name}/{fname}")
    print(f"Results Files Packing Done!({dir_name}/{fname})")


def json_load(path):
    with open(path,'r+') as load_f:
         config = json.load(load_f)
    return config


def json_save(data, path):
    with open(path,'w+') as dump_f:
        json.dump(data, dump_f, indent=4)
    # print(f"Optimization config save done ({path}).")


def time_formator(time):
    if time < 60:
        return time, "s"
    elif time <= 3600:
        return time / 60, "m"
    else:
        return time / 3600, "h"


def layout2grids(lb, ub, min_gs=5., mode="rank"):
    lb, ub = np.array(lb), np.array(ub)
    m, n = np.floor((ub - lb) / min_gs).astype(np.int)
    xs, ys = (ub - lb) / np.array([m, n])
    # x_bins, y_bins = np.linspace(lb[0], ub[0], m + 1), np.linspace(lb[1], ub[1], n + 1)
    # xs, ys = (x_bins[:-1] + x_bins[1:]) / 2, (y_bins[:-1] + y_bins[1:]) / 2
    xs, ys = np.arange(m + 1) * xs + lb[0], np.arange(n + 1) * ys + lb[1]

    if mode == 'rank':
        xs, ys = np.meshgrid(xs, ys[::-1])
        # grids = np.concatenate((xs.ravel()[None, :], ys.ravel()[None, :])).transpose(1, 0)
        grids = np.concatenate(
            (xs.transpose(1, 0)[None, :, :], ys.transpose(1, 0)[None, :, :]), axis=0)
        # print(grids.transpose(2, 1, 0).reshape((int(m + 1) * int(n + 1), 2)))
        return [int((m + 1) * (n + 1) - 1)], grids.transpose(0, 2, 1)
    else:
        grids = {}
        grids['x'], grids['y'] = xs, ys[::-1]
        return [int(m), int(n)], grids


def grids2layout(inds, grids, mode="rank"):
    grids = grids.transpose(1, 2, 0).reshape((grids.shape[1] * grids.shape[2], 2))
    if mode == 'rank':
        return grids[inds.astype(np.int), :]
    else:
        return np.concatenate((grids['x'][inds.astype(np.int)[:, 0]][:, None],
                               grids['y'][inds.astype(np.int)[:, 1]][:, None]), axis=1)


def mindist_calculation(layout, mindist=5., D_r=80.):
    metrix = euclidean_distances(layout)
    dist = np.min(metrix[np.tril_indices(metrix.shape[0], -1)])
    return mindist - dist


def pso_initial_particles(wtnum, num=30, D=80., layout=None):
    # np.random.seed(1234)
    baseline, spacing = fconfig.baseline_layout(wtnum, bounds=None,
                                            arrays=None, grids=None,
                                            spacing=True)
    xy_limits = (baseline / D)[[0, -1], :]
    xy_range = np.floor((np.array(spacing) / D - 5.) / 2)
    xmin, xmax, ymin, ymax = xy_limits[0, 0], xy_limits[1, 0], \
        xy_limits[1, 1], xy_limits[0, 1]
    layout = baseline / D if layout is None else layout
    init_n = layout.shape[0] if layout.ndim == 3 else 1
    xy_offset = np.concatenate((np.zeros((init_n, wtnum, 2)),
        np.random.uniform(-1, 1, (num - init_n, wtnum, 2))), axis=0)
    if init_n != 1:
        layout = np.concatenate(
            (layout, layout[0][None, :, :].repeat(num - init_n, axis=0)), axis=0)
    baseline = layout + (xy_range * xy_offset)
    baseline[:, :, 0] = np.clip(baseline[:, :, 0], xmin, xmax)
    baseline[:, :, 1] = np.clip(baseline[:, :, 1], ymin, ymax)
    return baseline.reshape((num, wtnum * 2))


def optimized_results_package(package):
    keys = ['config', 'layout', 'objval', 'fbest','favg','xbest', ]
    return {keys[i]:r for i, r in enumerate(package)}


def optimized_results_output(data, path, tag=None, stage=None):
    ptag = stage if stage is not None else ''
    ftag = f'_{tag}' if tag is not None else ''
    fname = f'{ptag}_results{ftag}'
    fpath = f'{path}/{fname}.json'
    with open(fpath,'w+') as dump_f:
        json.dump(data, dump_f, indent=4)
    print(f"{str.upper(ptag) + ' '}Optimization results save done ({fpath}).")


def optimized_results_combination(config, *args):
    temp = {'config':config,
            'layout':[r['layout'] for r in args],
            'objval':[r['objval'] for r in args],
            'fbest':[r['fbest'] for r in args],
            'favg':[r['favg'] for r in args],
            'xbest':[r['xbest'] for r in args],
            'stage':[len(r['fbest']) for r in args]}
    return temp
