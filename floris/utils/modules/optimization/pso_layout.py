import os
import numpy as np
import matplotlib.pyplot as plt

from floris.utils.tools import opt_ops as optops
from floris.utils.tools import farm_config as fconfig
from floris.utils.tools.skopt_pso import PSO as skpso
from floris.utils.visualization import wflo_opt as vwopt
from floris.utils.modules.optimization import wflo_layout as wflayout


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def skopt_layout(config):
    np.random.seed(3456)

    def wflo(layout):
        return wflayout.LayoutPower(config).run(
            np.array(layout).reshape(config['num'], 2) * 80.)[0]

    def min_dist(layout):
        layout = np.array(layout).reshape(config['num'], 2)
        return optops.mindist_calculation(layout)

    cons = (min_dist,)
    dim = config['num'] * 2
    lb = [0, 7.36] * config['num']
    ub = [63, 56.25] * config['num']
    vb = np.array(fconfig.baseline_layout(config['num'], grids=None, spacing=True)[1])
    vb = list( (vb / 80. - 0.) / 2) * config['num']
    pop, max_iter = 40, 120
    parts = optops.pso_initial_particles(config['num'], pop)

    pso = skpso(func=wflo,
                n_dim=dim,
                pop=pop,
                max_iter=max_iter,
                lb=lb, ub=ub, vb=vb,
                w=0.8, c1=0.5, c2=0.5,
                constraint_ueq=cons,
                verbose=True,
                initpts=parts,
                )

    pso.run()
    print('\nbest_y is', pso.gbest_y[0])
    optimized_layout = \
        np.array(pso.gbest_x, dtype=np.float).reshape((config['num'], 2)) 
    vwopt.wf_layout_plot(optimized_layout * 80.,
                        layout_name='pso_layout_25',)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pso.gbest_y_hist)
    plt.savefig('solution/pso_curve.png')
    plt.show()


class PSOptimizer(object):
    def __init__(self, config, pop=None, iters=None, seed=None,
                 w=0.5, c1=1.4, c2=1.4, precision=1e-3, N=70,
                 verbose=True):
        np.random.seed(seed)
        self.config = config
        self.wtnum = config['num']
        self.dim = 2 * self.wtnum
        self.lb = [0, 0] * self.wtnum
        self.ub = [63, 48.89] * self.wtnum
        self.vb = self.set_vb() * self.wtnum
        self.pop = pop or config['pop']
        self.iters = iters or config['maxg']
        self.trappedCount = N
        self.precision = precision
        self.verbose = verbose
        self.outdir = None
        self.result = None
        self.pso = None

        self.obj = self.obj_func(config)
        self.cons = self.obj_const(config)
        self.w, self.cp, self.cg = w, c1, c2

    def set_vb(self, sub_dist=0.):
        vb = np.array(fconfig.baseline_layout(self.wtnum, grids=None,
                                          spacing=True)[1])
        return list((vb / 80. - sub_dist) / 2 * 2)

    def init_parts(self, particles=None):
        return optops.pso_initial_particles(self.wtnum, self.pop,
                                          layout=particles)

    def obj_func(self, config):
        def wflo(layout):
            return wflayout.LayoutPower(config).run(
                np.array(layout).reshape(config['num'], 2) * 80.)[0]
        return wflo

    def obj_const(self, config):
        def min_dist(layout):
            layout = np.array(layout).reshape(config['num'], 2)
            return optops.mindist_calculation(layout)
        return (min_dist, )

    def init_optimizer(self, particles=None):
        self.inparts = self.init_parts(particles)
        self.pso = skpso(func=self.obj,
                         dim=self.dim,
                         pop=self.pop,
                         max_iter=self.iters,
                         lb=self.lb,
                         ub=self.ub,
                         vb=self.vb,
                         w=self.w,
                         c1=self.cp, c2=self.cg,
                         constraint_ueq=self.cons,
                         verbose=self.verbose,
                         initpts=self.inparts)
        return self.pso

    def run(self, outdir='solution', tag=None, output=False,
            analyse=False, **kwargs):
        self.init_optimizer(kwargs.get('particles', None))
        best_x, best_y = self.pso.run(**kwargs)

        package = [self.config, list(best_x), float(best_y),
                   self.pso.gbest_hist['f_best'],
                   self.pso.gbest_hist['f_avg'],
                   self.pso.gbest_hist['x_best']]
        self.result = optops.optimized_results_package(package)
        if output:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            tag = tag or self.config['tag']
            optops.optimized_results_output(
                self.result, outdir, tag, stage='pso')
            if analyse:
                self.stat(outdir, tag, **kwargs)

    def stat(self, outdir, tag=None, **kwargs):
        tag = f'_{tag}' if tag is not None else ''
        print(f'\nEvaluation times：{self.pso.evalsNum}')
        runtime = optops.time_formator(self.pso.passTime)
        print(f'Elapsed time {runtime[0]:.2f} {runtime[1]}')
        print(f'Optimal Levelized Cost：{self.pso.gbest_y[0]:.4f}')
        wflayout.analysis(path=outdir,
                      baseline="horns",
                      result=self.result,
                      config=self.config,
                      layout_output=True,
                      layout_name=f"pso_layout{tag}",
                      curve_output=True,
                      curve_name=f"pso_curve{tag}",
                      wd_output=True,
                      wd_name=f"pso_wds{tag}",
                      wt_output=True,
                      wt_name=f"pso_wts{tag}",
                      **kwargs)



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def skopt_test():

    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

    constraint_ueq = (
        lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2,)
    pso = skpso(func=demo_func, n_dim=3,
                pop=40, max_iter=150,
                lb=[0, -1, 0.5], ub=[1, 1, 1],
                w=0.8, c1=0.5, c2=0.5,
                constraint_ueq=constraint_ueq)
    pso.run()
    print('best_x is ', pso.gbest_x, '\nbest_y is', pso.gbest_y)
    plt.plot(pso.gbest_y_hist)
    plt.savefig('pictures/pso_test.png')
    plt.show()





if __name__ == "__main__":

    config = {
        "stage":1,
        "opt":'pso',
        "tag":'49',
        "pop": 30,
        # "pop": [10, 10],
        "maxg": 100,
        # "maxg": [10, 10],
        "grid": 5,
        "num": 49,
        "param": "horns",
        "wind": "single",
        "vbins": 3,
        "wbins": 15,
        "wdepth": "linear_x",
        "cost": "LCOE",
        "turb": 0.077,
        "wm": "Jensen",
        # "wm": "Bastankhah",
        "wsm": "SS",
        "tim": "Frandsen",
        # "tim": None,
    }
    
    outdir = 'output/21_7_01/Jen_single_49'
    
    PSOptimizer(config).run(outdir, output=True, analyse=True)
