import os
import sys
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt

from floris.utils.tools import opt_ops as optops
from floris.utils.tools import farm_config as fconfig
from floris.utils.tools import skopt_pso as skpso
from floris.utils.visual import wflo_opt as vwopt
from floris.utils.modules.optimization import wflo_layout as wflayout


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                               Genetic Algorithm                              #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class EALayout(ea.Problem):
    def __init__(self, configs, ):
        name = 'WindFarmLayout'
        self.config = configs
        self.wtnum = configs["num"]
        self.objFun = configs["cost"]
        self.grids = configs["grid"]
        self.strDim = 1
        objDim = 1
        objTypes = [1] if self.objFun != 'maxpower' else [-1]
        varDim = 2 * self.wtnum if self.grids is None else 1 * self.wtnum
        varTypes = [1] * varDim
        # lb = [0, 589]  ==> [0, 0]
        # ub = [5040, 4500]  ==> [5040, 3911]
        self.lb = [0, 0]
        self.ub = [63, 48.89]
        lb, ub = self.bounds()
        lbin = [1] * varDim
        ubin = [1] * varDim
        ea.Problem.__init__(self, name,
                            objDim, objTypes,
                            varDim, varTypes,
                            lb, ub, lbin, ubin)
        self.calculator = wflayout.LayoutPower(configs)

    def bounds(self):
        if self.grids is not None:
            lb, ub = [0], self.layout2grids(self.lb, self.ub)
        else:
            lb, ub = self.lb, self.ub
        return lb * self.wtnum, ub * self.wtnum

    def layout2grids(self, lb, ub, min_gs=5.):
        grid_ub, self.grids = optops.layout2grids(lb, ub, min_gs=min_gs, mode="rank")
        return grid_ub

    def grids2layout(self, inds):
        return optops.grids2layout(inds, self.grids)

    def minDist(self, layout, mind=5.):
        return optops.mindist_calculation(layout, mindist=mind)

    def aimFunc(self, pop, debug=True):
        Vars = pop.Phen if self.grids is not None else \
            pop.Phen.reshape(pop.Phen.shape[0], self.wtnum, 2)
        pop.ObjV = np.zeros((Vars.shape[0], self.M))
        pop.CV = np.zeros((Vars.shape[0], self.strDim))
        for i in range(Vars.shape[0]):
            layout = Vars[i, :] if self.grids is not None else Vars[i, :, :]
            if self.grids is not None:
                layout = self.grids2layout(layout)
            cost, _, _ = self.calculator.run(layout * 80.)
            pop.ObjV[i, 0] = cost
            pop.CV[i, :] = self.minDist(layout)


class EAOptimizer(object):
    def __init__(self, config, pop=None, maxg=None, seed=None,
                 precision=1e-3, N=50,):
        np.random.seed(seed)
        self.config = config
        self.problem = EALayout(config)
        self.Encoding = 'RI' if self.problem.grids is None else 'P'
        self.NIND = pop or config["pop"]
        self.MAXGEN = maxg or config["maxg"]
        self.precision = precision
        self.trappedCount = N
        self.solution = None
        self.outdir = None
        self.result = None
        self.algom = None

        if isinstance(self.NIND, list):
            self.population = []
            for i in range(len(self.NIND)):
                self.Field = ea.crtfld(self.Encoding, self.problem.varTypes,
                                       self.problem.ranges, self.problem.borders)
                self.population.append(ea.Population(self.Encoding, self.Field, self.NIND[i]))
        else:
            self.Field = ea.crtfld(self.Encoding, self.problem.varTypes,
                                self.problem.ranges, self.problem.borders)
            self.population = ea.Population(self.Encoding, self.Field, self.NIND)

    def run(self, outdir="solution", tag=None, prior=True, output=False,
            analyse=False, **kwargs):
        if isinstance(self.NIND, list):
            self.algom = ea.soea_multi_SEGA_templet(self.problem, self.population)
        else:
            self.algom = ea.soea_SEGA_templet(self.problem, self.population)
            self.algom.mutOper.Pm = 0.7  # 变异概率
        self.algom.trappedValue = self.precision
        self.algom.maxTrappedCount = self.trappedCount

        self.algom.MAXGEN = self.MAXGEN
        self.algom.logTras = 1
        self.algom.verbose = True
        self.algom.drawing = 0

        prophetPop = None
        if prior:
            baseline = optops.params_loader('horns').baseline(
                self.problem.wtnum, grids=self.problem.grids)
            if baseline is not None:
                prophetChrom = baseline if self.problem.grids is not None else \
                    (baseline / 80.).reshape((1, self.problem.wtnum * 2))
                prophetPop = ea.Population(self.Encoding, self.Field,
                                           prophetChrom.shape[0],
                                           prophetChrom)
                self.algom.call_aimFunc(prophetPop)
        [self.solution, self.population] = self.algom.run(prophetPop)
        if self.solution.sizes != 0:
            package = [self.config, list(self.solution.Phen[0]),
                       self.solution.ObjV[0][0],
                       self.algom.trace['f_best'],
                       self.algom.trace['f_avg'],
                       self.algom.trace['x_best']]
            self.result = optops.optimized_results_package(package)
            if output:
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                tag = tag or self.config['tag']
                optops.optimized_results_output(
                    self.result, outdir, tag, stage='ga')
                if analyse:
                    self.stat(outdir, tag, **kwargs)
            return True
        else:
            print('(GA) No feasible solution')
            return False

    def stat(self, outdir, tag=None, **kwargs):
        tag = f'_{tag}' if tag is not None else ''
        print(f'\nEvaluation times：{self.algom.evalsNum}')
        runtime = optops.time_formator(self.algom.passTime)
        print(f'Elapsed time {runtime[0]:.2f} {runtime[1]}')
        print(f'Optimal Levelized Cost：{self.solution.ObjV[0][0]}')
        wflayout.analysis(path=outdir,
                      baseline="horns",
                      result=self.result,
                      config=self.config,
                      layout_output=True,
                      layout_name=f"ga_layout{tag}",
                      curve_output=False,
                      curve_name=f"ga_curve{tag}",
                      wd_output=True,
                      wd_name=f"ga_wds{tag}",
                      wt_output=True,
                      wt_name=f"ga_wts{tag}",
                      **kwargs)


def assembled_optimizer():
    nums = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    tims = ["Crespo"]
    fnames = ["5_Jen_F", "10_Jen_F", "15_Jen_F", "20_Jen_F", "25_Jen_F", "30_Jen_F",
              "35_Jen_F", "40_Jen_F", "45_Jen_F", "50_Jen_F", "55_Jen_F", "60_Jen_F",]
    n = 0
    for tim in tims:
        for num in nums:
            config["tim"], config["num"] = tim, num
            EAOptimizer(config).run(fnames[n])
            # print(tim, num, fnames[n])
            n += 1



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                        Particle Swarm Optimization                           #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def PSOLayout(config):
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
    pso = skpso.PSO(func=wflo, n_dim=dim, pop=pop, max_iter=max_iter,
                    lb=lb, ub=ub, vb=vb, w=0.8, c1=0.5, c2=0.5,
                    constraint_ueq=cons, verbose=True, initpts=parts,)
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
        self.pso = skpso.PSO(func=self.obj,
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
        print(f'\nEvaluation times: {self.pso.evalsNum}')
        runtime = optops.time_formator(self.pso.passTime)
        print(f'Elapsed time {runtime[0]:.2f} {runtime[1]}')
        print(f'Optimal Levelized Cost: {self.pso.gbest_y[0]:.4f}')
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
#                               Genetic Algorithm                              #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class HybridOptimizer(object):
    def __init__(self, config, seed=None):
        np.random.seed(seed)
        self.config_check(config)
        self.config = config
        self.stageOpts = config['opt']
        self.stageNum = config['stage']
        self.pops = config['pop'] if isinstance(config['pop'], list) \
            else [config['pop']] * 2
        self.iters = config['maxg'] if isinstance(config['maxg'], list) \
            else [config['maxg']] * 2
        self.stageFlag = None
        self.eaopt = None
        self.psopt = None
        self.result = None

    def config_check(self, config):
        if config['stage'] == 1:
            if isinstance(config['opt'], list):
                if len(config['opt']) != 1:
                    raise RuntimeError('Invalid opt in config')
        elif config['stage'] == 2:
            if isinstance(config['opt'], str):
                raise RuntimeError('Invalid opt in config')
            if isinstance(config['opt'], list):
                if len(config['opt']) != 2:
                    raise RuntimeError('Invalid opt in config')
        else:
            raise RuntimeError('Invalid stage in config')

    def opt_init(self, ):
        return EAOptimizer(self.config, pop=self.pops[0], maxg=self.iters[0]), \
            PSOptimizer(self.config, pop=self.pops[1], iters=self.iters[1])

    def run(self, outdir="solution", tag=None, output=True, analyse=False,
            **kwargs):
        self.eaopt, self.psopt = self.opt_init()
        print('Stage One: GA ==>')
        if self.eaopt.run(outdir=outdir, tag=tag, **kwargs):
            layout = optops.grids2layout(self.eaopt.solution.Phen[0],
                                         self.eaopt.problem.grids)
            layout = layout[np.argsort(layout[:, 1]), :]
            print('\nStage two: PSO ==>')
            self.psopt.run(outdir=outdir, tag=tag, particles=layout, **kwargs)
            self.result = optops.optimized_results_combination(
                self.config, self.eaopt.result, self.psopt.result)
            if output:
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                tag = tag or self.config['tag']
                optops.optimized_results_output(
                    self.result, outdir, tag, stage='eapso')
                if analyse:
                    self.stat(outdir, tag, **kwargs)
                    self.curve(**kwargs)
        else:
            print('GA optimization did not find the solutions')

    def stat(self, outdir, tag=None, **kwargs):
        tag = f'_{tag}' if tag is not None else ''
        evalnum = self.eaopt.algom.evalsNum + self.psopt.pso.evalsNum
        print(f'\nTotal evaluation times：{evalnum}')
        runtime = optops.time_formator(
            self.eaopt.algom.passTime + self.psopt.pso.passTime)
        print(f'Total elapsed time: {runtime[0]:.2f} {runtime[1]}')
        print(f'Optimal Levelized Cost：{self.psopt.pso.gbest_y[0]:.4f}')
        wflayout.analysis(path=outdir,
                      baseline="horns",
                      result=self.psopt.result,
                      config=self.config,
                      layout_output=True,
                      layout_name=f"eapso_layout{tag}",
                      curve_output=True,
                      curve_name=f"eapso_curve{tag}",
                      wd_output=True,
                      wd_name=f"eapso_wds{tag}",
                      wt_output=True,
                      wt_name=f"eapso_wts{tag}",
                      **kwargs)


def assembled_optimizer(config):
    nums = [36, 49]
    # winds =['horns', 'average', 'single']
    # nums = [25, ]
    winds =['horns', ]
    for n in nums:
        for w in winds:
            config["num"], config["tag"] = n, str(n)
            fname = f"output/21_6_30/Jen_{n}_mos"
            try:
                print(f'\n******************* Case {n} with Mosetti**********************')
                HybridOptimizer(config).run(outdir=fname, analyse=True)
            except:
                continue


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def skopt_test():

    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

    constraint_ueq = (
        lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2,)
    pso = skpso.PSO(func=demo_func, n_dim=3, pop=40, max_iter=150,
                    lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5,
                    constraint_ueq=constraint_ueq)
    pso.run()
    print('best_x is ', pso.gbest_x, '\nbest_y is', pso.gbest_y)
    plt.plot(pso.gbest_y_hist)
    plt.savefig('pictures/pso_test.png')
    plt.show()



if __name__ == "__main__":

    config = {
        "stage":1,
        "opt":'ga',
        "tag":'25',
        "pop": 5,
        # "pop": [10, 10],
        "maxg": 5,
        # "maxg": [10, 10],
        "grid": 5,
        "num": 25,
        "param": "horns",
        "wind": "horns",
        "vbins": 3,
        "wbins": 15,
        "wdepth": "linear_x",
        "cost": "LCOE",
        "turb": 0.077,
        "wm": "Jensen",
        # "wm": "Bastankhah",
        "wsm": "SS",
        # "tim": "Frandsen",
        "tim": None,
    }
    # EAOptimizer(config).run(analyse=True)
    print(sys.path)

    # config = {
    #     "stage":1,
    #     "opt":'pso',
    #     "tag":'49',
    #     "pop": 30,
    #     # "pop": [10, 10],
    #     "maxg": 100,
    #     # "maxg": [10, 10],
    #     "grid": 5,
    #     "num": 49,
    #     "param": "horns",
    #     "wind": "single",
    #     "vbins": 3,
    #     "wbins": 15,
    #     "wdepth": "linear_x",
    #     "cost": "LCOE",
    #     "turb": 0.077,
    #     "wm": "Jensen",
    #     # "wm": "Bastankhah",
    #     "wsm": "SS",
    #     "tim": "Frandsen",
    #     # "tim": None,
    # }
    # outdir = 'output/21_7_01/Jen_single_49'
    # PSOptimizer(config).run(outdir, output=True, analyse=True)

    # config = {
    #     "stage":2,
    #     "opt":['ga', 'pso'],
    #     "tag":'25',
    #     "pop": 50,
    #     # "pop": [20, 20],
    #     # "maxg": 5,
    #     "maxg": [40, 150],
    #     "grid": 7,
    #     "num": 25,
    #     "param": "horns",
    #     "wind": "single",
    #     "vbins": 3,
    #     "wbins": 15,
    #     "wdepth": "linear_x",
    #     "cost": "LCOE",
    #     "turb": 0.077,
    #     # "wm": "Jensen",
    #     "wm": "Bastankhah",
    #     "wsm": "SS",
    #     "tim": "Frandsen",
    #     # "tim": None,
    # }
    # HybridOptimizer(config).run(
    #     outdir='output/21_7_01/BP_single_25', output=True, analyse=True)
    # # assembled_optimizer(config)
