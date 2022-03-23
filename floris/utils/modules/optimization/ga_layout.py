import os
import sys
import geatpy as ea
import numpy as np

from floris.utils.tools import opt_ops as optops
from floris.utils.modules.optimization import wflo_layout as wflayout


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
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
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #





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
    
