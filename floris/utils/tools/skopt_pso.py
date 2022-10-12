from time import time
import numpy as np
from sko.tools import func_transformer
from sko.base import SkoBase


class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint. Note: not available yet.
    constraint_ueq : tuple
        unequal constraint
    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5,
                 w=0.8, c1=0.5, c2=0.5, constraint_eq=tuple(), constraint_ueq=tuple(),
                 verbose=False, dim=None, initpts=None, vb=None, **kwargs):

        n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func)
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not
        self.currentIter = 0  # 初始为第0代
        self.evalsNum = 0  # 初始化评价次数为0

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        self.X = np.random.uniform(low=self.lb, high=self.ub,
                                   size=(self.pop, self.n_dim)) if initpts is None else initpts
        v_high = self.ub - self.lb if vb is None else np.array(vb)
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.gbest_hist = {"f_best": [], "f_avg": [], "x_best": []} # gbest_y of every iteration
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        # self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.max(self.pbest_y)  # global best y for all particles
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        
        self.log = {'gen': [], 'eval': []}
        self.passTime = 0  # 初始化passTime
        self.evalTime = 0  # 初始化evalTime
        self.timeSlot = time()   # 开始计时

    def check_constraint(self, x):
        # gather all unequal constraint functions
        # for constraint_func in self.constraint_ueq:
        #     # print(x.reshape((25, 2)))
        #     # print(constraint_func(x))
        #     if constraint_func(x) > 0:
        #         return False
        return all(constraint_func(x) <= 0 for constraint_func in self.constraint_ueq)

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)
        # print(self.V[0])

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)
        # print(self.X[0].reshape((25, 2)))

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)
    
    def logging(self):

        self.passTime += time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
        if len(self.log['gen']) == 0:  # 初始化log的各个键值
            self.log['f_opt'] = []
            self.log['f_max'] = []
            self.log['f_avg'] = []
            self.log['f_min'] = []
            self.log['f_std'] = []
            self.log['runtime'] = []
        self.log['gen'].append(self.currentIter)
        self.log['eval'].append(self.evalsNum)  # 记录评价次数
        self.log['f_opt'].append(self.gbest_y[0])  # 记录算法所找到的最优个体的目标函数值
        self.log['f_max'].append(np.max(self.pbest_y))
        self.log['f_avg'].append(np.mean(self.pbest_y))
        self.log['f_min'].append(np.min(self.pbest_y))
        self.log['f_std'].append(np.std(self.pbest_y))
        self.log['runtime'].append(self.evalTime)
        self.timeSlot = time()  # 更新时间戳

    def display(self):
        self.passTime += time() - self.timeSlot  # 更新用时记录，不计算display()的耗时
        headers = []
        widths = []
        values = []
        for key in self.log.keys():
            # 设置单元格宽度
            if key == 'gen':
                width = max(3, len(str(self.max_iter - 1)))  # 因为字符串'gen'长度为3，所以最小要设置长度为3
            elif key == 'eval':
                width = 8  # 因为字符串'eval'长度为4，所以最小要设置长度为4
            else:
                width = 13  # 预留13位显示长度，若数值过大，表格将无法对齐，此时若要让表格对齐，需要自定义算法模板重写该函数
            headers.append(key)
            widths.append(width)
            value = self.log[key][-1] if len(self.log[key]) != 0 else "-"
            if isinstance(value, float):
                if key == 'f_std':
                    values.append("%.5e" % value)
                else:
                    values.append("%.4f" % value)  # 格式化浮点数，输出时只保留至小数点后5位
            else:
                values.append(value)
        if len(self.log['gen']) == 1:  # 打印表头
            header_regex = '|'.join(['{}'] * len(headers))
            header_str = header_regex.format(*[str(key).center(width) for key, width in zip(headers, widths)])
            print("=" * len(header_str))
            print(header_str)
            print("-" * len(header_str))
        if len(self.log['gen']) != 0:  # 打印表格最后一行
            value_regex = '|'.join(['{}'] * len(values))
            value_str = value_regex.format(*[str(value).center(width) for value, width in zip(values, widths)])
            print(value_str)
        self.timeSlot = time()  # 更新时间戳
    
    def run(self, max_iter=None, precision=1e-4, N=30,
            **kwargs):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.currentIter = iter_num + 1
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0

            self.evalsNum += self.pop
            self.gbest_hist['f_best'].append(float(self.gbest_y))
            self.gbest_hist['f_avg'].append(float(np.mean(self.pbest_y)))
            self.gbest_hist['x_best'].append(list(self.gbest_x))
            self.evalTime = time() - self.timeSlot
            self.passTime += self.evalTime

            self.logging()
            if self.verbose:
                self.display()

            self.timeSlot = time()

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run


