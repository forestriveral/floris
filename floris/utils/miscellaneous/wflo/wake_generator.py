import os
import sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)
import numpy as np
from scipy import integrate



class BPWakeGenerator(object):
    #  (非)偏航状态下Bastankhah-Porté-Agel模型
    def __init__(self, vel, turb, C_thrust, offset, D_rotor, z_hub=None):
        self.velocity = vel         # 风速
        self.turbulence = turb      # 湍流度
        self.C_thrust = C_thrust    # 推力系数
        self.offset = offset        # 偏航角
        self.D_rotor = D_rotor      # 风轮直径
        
        self.k_star = 0.3837 * self.turbulence + 0.003678   # 尾流边界斜率
        self.epsilon = 0.2 * np.sqrt(
            (1. + np.sqrt(1 - self.C_thrust)) / (2. * np.sqrt(1 - self.C_thrust)))
    
    def wake_sigma_Dr(self, x_D):   # 尾流宽度
        return self.k_star * x_D + self.epsilon
    
    def wake_constant(self, sigma_Dr):    # 表达式系数
        A = 1. - np.sqrt(1. - self.C_thrust / (8 * sigma_Dr**2)) \
            if self.C_thrust / (8 * sigma_Dr**2) <= 1. else 1.
        B = - 0.5 /  (sigma_Dr**2)
        return A, B

    def wake_offset(self, x_D):   # 尾流偏移量
        theta = self.offset / 360 * 2 * np.pi
        theta_func = lambda x_D: np.tan(np.cos(theta)**2 * np.sin(theta) * \
            self.C_thrust * 0.5 * (1 + 0.09 * x_D)**-2)
        offset =  integrate.quad(theta_func, 0, x_D)[0] * self.D_rotor
        return offset
    
    def wake_mesh(self, xb, yb, zb):   # 尾流场网格划分
        # xb = x axis boundary; yb = y axis boundary; zb = z axis boundary
        # xb: (min, max, num); yb: (min, max, num); zb: (min, max, num)
        xs = np.linspace(xb[0], xb[1], xb[2])
        zs, ys = np.meshgrid(np.linspace(zb[0], zb[1], zb[2]),
                             np.linspace(yb[0], yb[1], yb[2]))
        return xs, ys, zs
    
    def wake_section(self, y_D, z_D, A, B):   # 特定距离下尾流分布
        deficit = A * np.exp(B * (z_D**2 + y_D**2))
        return self.velocity * (1 - deficit)
    
    def wake_field(self, xb=(1, 10, 10), yb=(-2, 2, 5), zb=(-1.5, 1.5, 5)):  # 尾流场数据计算
        xs, ys, zs = self.wake_mesh(xb, yb, zb)
        wake_flow = np.zeros((xb[2], yb[2], zb[2]))
        for i, x_D in enumerate(xs):
            A_i, B_i = self.wake_constant(self.wake_sigma_Dr(x_D))
            d_D = self.wake_offset(x_D) / self.D_rotor
            wake_flow[i, :, :] = \
                np.vectorize(self.wake_section)(ys + d_D, zs, A_i, B_i)[:,::-1].T
        # 尾流场数组维度  0：x方向(1->10);  1:z方向(-1.5->1.5);  2:y方向(-2->2)
        return wake_flow



if __name__ == "__main__":
    
    # 来流风况和风机参数
    vel = 8.           # 风速   5. ~ 25.
    turb = 0.077       # 湍流度  0.077 ~ 0.150
    C_thrust = 0.75    # 推力系数  0 ~ 1
    offset = 30.       # 偏航角    0 ~ 30
    D_rotor = 80.      # 风轮直径
    z_hub = 70.        # 轮毂高度(可暂时不用)
    # 尾流场数据点范围(以叶轮中心为原点)
    xb = (1, 10, 10)       # x轴=沿风速方向        1D ~ 10D     10个点
    yb = (-2, 2, 5)        # y轴=水平面垂直方向    -2D ~ 2D      5个点
    zb = (-1.5, 1.5, 5)    # z轴=自然高度风向     -1.5D ~ 1.5D   5个点
    
    wake = BPWakeGenerator(vel, turb, C_thrust, offset, D_rotor, z_hub).wake_field(xb, yb, zb)
    print(wake[5, :, :])   #  6D 尾流截面速度分布