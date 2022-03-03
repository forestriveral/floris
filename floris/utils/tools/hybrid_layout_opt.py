import os
import numpy as np

from floris.utils.modules import wflo_layout as wflayout
from floris.utils.tools import opt_ops as optops
from floris.utils.tools import ga_layout_opt as EAOptimizer
from floris.utils.tools import pso_layout_opt as PSOptimizer

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
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




if __name__ == "__main__":
    
    config = {
        "stage":2,
        "opt":['ga', 'pso'],
        "tag":'25',
        "pop": 50,
        # "pop": [20, 20],
        # "maxg": 5,
        "maxg": [40, 150],
        "grid": 7,
        "num": 25,
        "param": "horns",
        "wind": "single",
        "vbins": 3,
        "wbins": 15,
        "wdepth": "linear_x",
        "cost": "LCOE",
        "turb": 0.077,
        # "wm": "Jensen",
        "wm": "Bastankhah",
        "wsm": "SS",
        "tim": "Frandsen",
        # "tim": None,
    }
    
    
    HybridOptimizer(config).run(
        outdir='output/21_7_01/BP_single_25', output=True, analyse=True)
    # assembled_optimizer(config)