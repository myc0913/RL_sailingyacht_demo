import numpy as np
import pygame
from typing import Optional
import globals

import math
import random
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp
from scipy.integrate import ode

FPS = 60

class SailboatwindV1(gym.Env,EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_Fps": FPS,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            ):

        EzPickle.__init__(
            self,
            render_mode,) 

        # 定义常量
        super(SailboatwindV1, self).__init__()
        # global current_state, current_action, previous_state, previous_action, current_time, total_time
        # global path, end_pos, w0, var_v, var_d, w_change_v, w_change_d, w_last_v, w_last_d, w_ini_v, w_ini_d
        
        #仿真时间相关
        self.dt = 0.05
        self.step_per_action = int(1.0 / self.dt)   

        self.par = {
            'm': 25900,                         # (kg),mass of the vehicle
            'Ixx': 133690,          
            'Izz': 24760,      
            'Ixz': 2180,                        # moment of inertia
            'a11': 970,   
            'a22': 17430, 
            'a44': 106500,
            'a66': 101650,
            'a24': -13160,
            'a26': -6190, 
            'a46': 4730,                        # (kg),added mass coef.
            'rho_a': 1.2,                       # (kg/m^3), air density
            'As': 170,                          # (m^2), sail area
            'h0': 0.0005,                       # (m), roughness height
            'h1': 11.58,                        # (m), reference height
            'z_s': -11.58,                      # (m), (x,y,z) is the CoE
            'xs': 0,
            'ys': 0,
            'zs': -11.58,                       # (m), (x,y,z) is the CoE
            'Xce': 0.6,                         # (m), distance along the mast to the CoE
            'Xm': 0.3,                          # (m), x-coordinate of the mast 
            'rho_w': 1025,                      # (kg/m^3), water density
            'Ar': 1.17,                         # (m^2), rudder area
            'd_r': 1.9,                         # rudder draft
            'zeta_r': 0.8,                      # rudder efficiency
            'x_r': -8.2,
            'z_r': -0.78,                       # (m), (x,y,z) is the CoE
            'xr': -8.2,
            'yr': 0,
            'zr': -0.78,                        # (m), (x,y,z) is the CoE
            'Ak': 8.7,                          # (m^2), keel area
            'd_k': 2.49,                        # keel draft
            'zeta_k': 0.7,                      # keel efficiency
            'x_k': 0,
            'z_k': -0.58,                       # (m), (x,y,z) is the CoE
            'xk': 0,
            'yk': 0,
            'zk': -0.58,                        # (m), (x,y,z) is the CoE
            'x_h': 0,
            'z_h': -1.18,                       # (m), (x,y,z) is the CoE
            'xh': 0,
            'yh': 0,
            'zh': -1.18,                        # (m), (x,y,z) is the CoE
            'w_c': 60000,                       # (N), crew weight 20000
            'x_c': -8,                          # (m), crew position
            'y_bm': 3.6,                        # (m), yacht beam
            'a': -5.89,
            'b': 8160,
            'c': 120000,
            'd': 50000
        }

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-np.pi/18, -np.pi/6]), high=np.array([np.pi/18, np.pi/6]), dtype=np.float32)

        self.state = None
        self.screen = None

        self.reset()

        self.viewer = None
        self.clock = pygame.time.Clock()


    def step(self, action):
        # global current_state, current_action, previous_state, previous_action,current_time, total_time
        # global path, end_pos, w0, var_v, var_d, w_change_v, w_change_d, w_last_v, w_last_d, w_ini_v, w_ini_d 
        t = 0
        tf = 1
        dt = 0.05
        tol = 1e-6
        flag = True
        sail_error = False
        rudder_error = False
        #采样时间与仿真时间
        if flag:
            previous_state = np.copy(self.state[0:11])
            flag = False
        globals.current_state = self.state[0:11]
        globals.current_action = action
        globals.state[13] = np.copy(globals.w_change_d)
        globals.state[14] = np.copy(globals.w_change_v)

        globals.path.append((self.state[0], self.state[1]))

        def state_derivatives(state, action):
            x, y, phi, psi, u, v, p, r = state[0:8]
            nu = np.array([[u], [v], [p], [r]])
            delta_s, delta_r = action[0:2]
            y_w = 0
            sail = np.copy(globals.current_state[8]) + delta_s // (np.pi/180) * (np.pi/180)
            rudder = np.copy(globals.current_state[9]) + delta_r // (np.pi/180) * (np.pi/180)

            if sail>= np.pi/2:
                sail = np.pi/2
            elif sail<= -np.pi/2:
                sail = -np.pi/2

            if rudder>= np.pi/6:
                rudder = np.pi/6
            elif rudder<= -np.pi/6:
                rudder = -np.pi/6
            
            M_RB = np.array([
                [self.par['m'], 0, 0, 0],
                [0, self.par['m'], 0, 0],
                [0, 0, self.par['Ixx'], -self.par['Ixz']],
                [0, 0, -self.par['Ixz'], self.par['Izz']]
            ])

            C_RB = np.array([
                [0,-self.par['m']*r,0,0],
                [self.par['m']*r,0,0,0],
                [0,0,0,0],
                [0,0,0,0]
            ])

            M_A = np.array([
                [self.par['a11'], 0, 0, 0],
                [0, self.par['a22'], self.par['a24'], self.par['a26']],
                [0, self.par['a24'], self.par['a44'], self.par['a46']],
                [0, self.par['a26'], self.par['a46'], self.par['a66']]
            ])

            C_A = np.array([
                [0,0,0,-self.par['a22']*nu[1,0] - self.par['a24']*nu[2,0] - self.par['a26']*nu[3,0]],
                [0, 0, 0, self.par['a11']*nu[0,0]],
                [0, 0, 0, 0],
                [self.par['a22']*nu[1,0] + self.par['a24']*nu[2,0] + self.par['a26']*nu[3,0], -self.par['a11']*nu[0,0], 0, 0]
                ])
            
            M = M_RB + M_A

            v_t = np.array([
                [globals.w_change_v*np.cos(globals.w_change_d)], 
                [globals.w_change_v*np.sin(globals.w_change_d)],
                [0]
                  ])
            
            numerator = abs(self.par['z_s']) * np.cos(phi) / self.par['h0']
            # numerator = np.where(numerator <= 0, 1e-10, numerator)

            v_tw = np.log(numerator)/np.log(self.par['h1']/self.par['h0'])*v_t
            R1 = np.array([
                [np.cos(-psi), -np.sin(-psi), 0],
                [np.sin(-psi), np.cos(-psi), 0],
                [0, 0, 1]
            ])
            R2 = np.array([
                [1, 0, 0],
                [0, np.cos(-phi), -np.sin(-phi)],
                [0, np.sin(-phi), np.cos(-phi)]
            ])

            v_tb = R2 @ R1 @ v_tw
            V_in = np.array([[u], [v], [0]])

            cross_1 = np.array([[p],[0],[r]])
            cross_2 = np.array([[self.par['xs']],[self.par['ys']],[self.par['zs']]])
            v_awb = v_tb - V_in - np.cross(cross_1.flatten(),cross_2.flatten())
            v_awu = v_awb[0][0]
            v_awv = v_awb[1][0]
            alpha_aw = np.arctan2(v_awv,-v_awu)

            alpha_as = alpha_aw - sail

            Cls,Cds = sailcoef(alpha_as)
            Ls = 0.5 * self.par['rho_a'] * (v_awu**2 + v_awv**2) * self.par['As'] * Cls
            Ds = 0.5 * self.par['rho_a'] * (v_awu**2 + v_awv**2) * self.par['As'] * Cds
            tau_sail = np.array([
                [Ls * np.sin(alpha_aw) - Ds * np.cos(alpha_aw)],
                [Ls * np.cos(alpha_aw) + Ds * np.sin(alpha_aw)],
                [-(Ls * np.cos(alpha_aw) + Ds * np.sin(alpha_aw)) * self.par['zs']],
                [-(Ls * np.sin(alpha_aw) - Ds * np.cos(alpha_aw)) * self.par['Xce'] * np.sin(sail) + (Ls * np.cos(alpha_aw) + Ds * np.sin(alpha_aw)) * (self.par['Xm'] - self.par['Xce'] * np.cos(sail))]
            ])

            Mzs = tau_sail[3]

            v_aru = -u+r*self.par['yr']
            v_arv = -v-r*self.par['xr']+p*self.par['zr']
            alpha_ar = np.arctan2(v_arv,-v_aru)
            alpha_a = alpha_ar - rudder
            Clr, Cdr = ruddercoef(alpha_a)
            Lr = 0.5*self.par['rho_w']*self.par['Ar']*(v_aru**2+v_arv**2)*Clr
            Dr = 0.5*self.par['rho_w']*self.par['Ar']*(v_aru**2+v_arv**2)*Cdr

            tau_rudder = np.array([
                [Lr*np.sin(alpha_ar)-Dr*np.cos(alpha_ar)],
                [Lr*np.cos(alpha_ar)+Dr*np.sin(alpha_ar)],
                [-(Lr*np.cos(alpha_ar)+Dr*np.sin(alpha_ar))*self.par['zr']],
                [(Lr*np.cos(alpha_ar)+Dr*np.sin(alpha_ar))*self.par['xr']]
            ])
            
            Mzr = tau_rudder[3]

            tau = tau_sail + tau_rudder

            v_aku = -u+r*self.par['yk']
            v_akv = -v-r*self.par['xk']+p*self.par['zk']
            alpha_ak = np.arctan2(v_akv,-v_aku)
            alpha_e = alpha_ak
            clk, cdk = keelcoef(alpha_e)
            cdk = cdk + (clk**2)*self.par['Ak']/(np.pi*2*self.par['zeta_k']*(self.par['d_k']**2))

            Lk = 0.5 * self.par['rho_w'] * self.par['Ak'] * (v_aku**2 + v_akv**2) * clk
            Dk = 0.5 * self.par['rho_w'] * self.par['Ak'] * (v_aku**2 + v_akv**2) * cdk

            # 计算 D_keel
            D_keel = np.array([
                [-Lk * np.sin(alpha_ak) + Dk * np.cos(alpha_ak)],
                [-Lk * np.cos(alpha_ak) - Dk * np.sin(alpha_ak)],
                [-(-Lk * np.cos(alpha_ak) - Dk * np.sin(alpha_ak)) * self.par['zk']],
                [-(Lk * np.cos(alpha_ak) + Dk * np.sin(alpha_ak)) * self.par['xk']]
            ])

            v_ahu = -u + r * self.par['yh']
            v_ahv = (-v - r * self.par['xh'] + p * self.par['zh'])/np.cos(phi)
            v_ah = np.sqrt(v_aku**2 + v_akv**2)
            alpha_ah = np.arctan2(v_ahv, -v_ahu)
            Frh = resistancehull(v_ah)
            
            # 计算 D_hull
            D_hull = np.array([
                [Frh * np.cos(alpha_ah)],
                [-Frh * np.sin(alpha_ah) * np.cos(phi)],
                [Frh * np.sin(alpha_ah) * np.cos(phi) * self.par['zh']],
                [-Frh * np.sin(alpha_ah) * np.cos(phi) * self.par['xh']]
            ])

            # 计算 heel 和 yaw 的阻尼力
            J = np.array([
                [np.cos(psi), -np.sin(psi) * np.cos(phi), 0, 0],
                [np.sin(psi), np.cos(psi) * np.cos(phi), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.cos(phi)]
            ])
            eta_dot = J @ nu
            phi_dot = eta_dot[2,0]
            psi_dot = eta_dot[3,0]

            D_heelandyaw = np.array([
                [0],
                [0],
                [self.par['c'] * phi_dot * abs(phi_dot)],
                [self.par['d'] * psi_dot * abs(psi_dot) * np.cos(phi)]
            ])

            # 计算总阻尼向量 D
            D = D_keel + D_hull + D_heelandyaw

            # 计算复原力矩和内部移动质量系统（即横向重量）
            phi_deg = phi * 180 / np.pi
            M_xw = -y_w * self.par['w_c'] * self.par['y_bm'] * np.cos(phi)
            M_zw = -y_w * self.par['w_c'] * self.par['x_c'] * np.sin(abs(phi))
            G = np.array([
                [0],
                [0],
                [self.par['a'] * phi_deg**2 + self.par['b'] * phi_deg + M_xw],
                [M_zw]
            ])

            # 计算 nu_dot
            nu_dot = -np.linalg.solve(M, C_RB @ nu + C_A @ nu) - np.linalg.solve(M, D) - np.linalg.solve(M, G) + np.linalg.solve(M, tau)

            # 输出状态导数扩展与帆角
            X_dot_ext = np.concatenate([eta_dot.flatten(), nu_dot.flatten()])

            return X_dot_ext


        def sailcoef(attack):
            # 生成升力/阻力系数的查找表并使用插值计算 Cls/Cds

            # 查找表
            xdata = np.linspace(-np.pi, np.pi, 73)  # 每 5 度
            xdata = xdata / np.pi * 180

            # 升力曲线
            yldata = np.concatenate([
                np.flip([0, 0.15, 0.32, 0.48, 0.7, 0.94, 1.15, 1.3, 1.28, 1.15, 1.1, 1.05, 1, 0.9, 0.82, 0.72, 0.68, 0.56, 0.48, 0.32, 0.21, 0.08, -0.06, -0.18, -0.3, -0.4, -0.53, -0.64, -0.72, -0.84, -0.95, -1.04, -1.1, -1.14, -1.08, -0.76, 0]) * (-1),
                [0.15, 0.32, 0.48, 0.7, 0.94, 1.15, 1.3, 1.28, 1.15, 1.1, 1.05, 1, 0.9, 0.82, 0.72, 0.68, 0.56, 0.48, 0.32, 0.21, 0.08, -0.06, -0.18, -0.3, -0.4, -0.53, -0.64, -0.72, -0.84, -0.95, -1.04, -1.1, -1.14, -1.08, -0.76, 0]
            ])

            # 阻力曲线
            yddata = np.concatenate([
                np.flip([0.1, 0.12, 0.14, 0.16, 0.19, 0.26, 0.35, 0.46, 0.54, 0.62, 0.7, 0.78, 0.9, 0.98, 1.04, 1.08, 1.16, 1.2, 1.24, 1.26, 1.28, 1.34, 1.36, 1.37, 1.33, 1.31, 1.28, 1.26, 1.25, 1.2, 1.1, 1.04, 0.88, 0.8, 0.64, 0.38, 0.1]),
                [0.12, 0.14, 0.16, 0.19, 0.26, 0.35, 0.46, 0.54, 0.62, 0.7, 0.78, 0.9, 0.98, 1.04, 1.08, 1.16, 1.2, 1.24, 1.26, 1.28, 1.34, 1.36, 1.37, 1.33, 1.31, 1.28, 1.26, 1.25, 1.2, 1.1, 1.04, 0.88, 0.8, 0.64, 0.38, 0.1]
            ])

            # 将输入的攻角调整到 [-pi, pi] 范围内
            if attack > np.pi:
                attack = (attack + np.pi) % (2 * np.pi) - np.pi
            elif attack < -np.pi:
                attack = (attack - np.pi) % (-2 * np.pi) + np.pi

            # 插值计算
            attack = attack / np.pi * 180
            Cls = interp1d(xdata, yldata, kind='cubic')(attack)
            Cds = interp1d(xdata, yddata, kind='cubic')(attack)

            return Cls, Cds
        
        def ruddercoef(attack):
            # 生成升力/阻力系数的查找表并使用插值计算 Clr/Cdr

            # 查找表
            xdata = np.linspace(-np.pi, np.pi, 73)  # 每 5 度
            xdata = xdata / np.pi * 180

            # 升力曲线
            yl = np.flip([0, 0.42, 0.73, 0.95, 1.1, 1.165, 1.18, 1.155, 1.12, 1.065, 1, 0.92, 0.83, 0.72, 0.62, 0.48, 0.33, 0.16]) * (-1)
            yldata = np.concatenate([
                np.flip([0, 0.42, 0.73, 0.95, 1.1, 1.165, 1.18, 1.155, 1.12, 1.065, 1, 0.92, 0.83, 0.72, 0.62, 0.48, 0.33, 0.16, 0, *yl]) * (-1),
                [0.42, 0.73, 0.95, 1.1, 1.165, 1.18, 1.155, 1.12, 1.065, 1, 0.92, 0.83, 0.72, 0.62, 0.48, 0.33, 0.16, 0, *yl]
            ])

            # 阻力曲线
            yd = np.flip([0, 0.03, 0.06, 0.1, 0.17, 0.3, 0.48, 0.74, 0.98, 1.18, 1.34, 1.5, 1.65, 1.76, 1.89, 1.97, 2.01, 2.05])
            yddata = np.concatenate([
                np.flip([0, 0.03, 0.06, 0.1, 0.17, 0.3, 0.48, 0.74, 0.98, 1.18, 1.34, 1.5, 1.65, 1.76, 1.89, 1.97, 2.01, 2.05, 2.08, *yd]),
                [0.03, 0.06, 0.1, 0.17, 0.3, 0.48, 0.74, 0.98, 1.18, 1.34, 1.5, 1.65, 1.76, 1.89, 1.97, 2.01, 2.05, 2.08, *yd]
            ])

            # 将输入的攻角调整到 [-pi, pi] 范围内
            if attack > np.pi:
                attack = (attack + np.pi) % (2 * np.pi) - np.pi
            elif attack < -np.pi:
                attack = (attack - np.pi) % (-2 * np.pi) + np.pi

            # 插值计算
            attack = attack / np.pi * 180
            Clr = interp1d(xdata, yldata, kind='cubic')(attack)
            Cdr = interp1d(xdata, yddata, kind='cubic')(attack)

            return Clr, Cdr
        
        def keelcoef(attack):
            # 生成升力/阻力系数的查找表并使用插值计算 Clk/Cdk

            # 查找表
            xdata = np.linspace(-np.pi, np.pi, 73)  # 每 5 度
            xdata = xdata / np.pi * 180

            # 升力曲线
            yl = np.flip([0, 0.425, 0.74, 0.94, 1.1, 1.17, 1.19, 1.16, 1.12, 1.07, 0.99, 0.92, 0.84, 0.74, 0.63, 0.49, 0.345, 0.185]) * (-1)
            yldata = np.concatenate([
                np.flip([0, 0.425, 0.74, 0.94, 1.1, 1.17, 1.19, 1.16, 1.12, 1.07, 0.99, 0.92, 0.84, 0.74, 0.63, 0.49, 0.345, 0.185, 0, *yl]) * (-1),
                [0.425, 0.74, 0.94, 1.1, 1.17, 1.19, 1.16, 1.12, 1.07, 0.99, 0.92, 0.84, 0.74, 0.63, 0.49, 0.345, 0.185, 0, *yl]
            ])

            # 阻力曲线
            yd = np.flip([0, 0.04, 0.07, 0.1, 0.17, 0.3, 0.49, 0.76, 0.98, 1.19, 1.34, 1.5, 1.65, 1.77, 1.88, 1.96, 2.01, 2.05])
            yddata = np.concatenate([
                np.flip([0, 0.04, 0.07, 0.1, 0.17, 0.3, 0.49, 0.76, 0.98, 1.19, 1.34, 1.5, 1.65, 1.77, 1.88, 1.96, 2.01, 2.05, 2.09, *yd]),
                [0.04, 0.07, 0.1, 0.17, 0.3, 0.49, 0.76, 0.98, 1.19, 1.34, 1.5, 1.65, 1.77, 1.88, 1.96, 2.01, 2.05, 2.09, *yd]
            ])

            # 将输入的攻角调整到 [-pi, pi] 范围内
            if attack > np.pi:
                attack = (attack + np.pi) % (2 * np.pi) - np.pi
            elif attack < -np.pi:
                attack = (attack - np.pi) % (-2 * np.pi) + np.pi

            # 插值计算
            attack = attack / np.pi * 180
            Clk = interp1d(xdata, yldata, kind='cubic')(attack)
            Cdk = interp1d(xdata, yddata, kind='cubic')(attack)

            return Clk, Cdk

        def resistancehull(alpha_a):
            # 生成船体阻力的查找表并使用插值计算 F

            # 查找表
            xdata = np.linspace(0, 6, 13)  # 每 0.5 m/s
            ydata = np.array([0, 0.15, 0.35, 0.5, 0.675, 0.825, 1.175, 1.4, 2, 4.85, 9.85, 18.46, 27.5]) * 1000

            # 插值计算
            if alpha_a <= 6:
                F = interp1d(xdata, ydata, kind='cubic')(alpha_a)
            else:
                F = interp1d(xdata, ydata, kind='cubic', fill_value='extrapolate')(alpha_a)

            return F   
                
        # for _ in range(self.step_per_action):

        #     # globals.current_state += state_derivatives(globals.current_state, action) * self.dt            
        #     k1 = state_derivatives(globals.current_state, action)
        #     k2 = state_derivatives(globals.current_state + k1 / 2 * self.dt, action)
        #     k3 = state_derivatives(globals.current_state + 0.75 * k2 * self.dt, action)
        #     globals.current_state += self.dt * (k1 * 2/9 + k2 * 1/3 + k3 * 4/9)
        #     self.state = globals.current_state
        #     globals.current_time += self.dt
        #     truncated = globals.current_time >= globals.total_time

        #     if truncated:
        #         break

        def distance(x1, y1, x2, y2):
        # 计算两点之间的距离
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        def projection_point(x, y, slope, intercept):
            # 计算点 (x, y) 在直线上的投影点
            if slope is None:
                return 0, y  # 垂直线的投影点
            x_proj = (x + slope * y) / (1 + slope ** 2)
            y_proj = slope * x_proj
            return x_proj, y_proj
        
        def projection_length(x1, y1, x2, y2):
            slope, intercept = globals.slope, globals.intercept
            end_x, end_y = globals.state[11:13]
            x1_proj, y1_proj = projection_point(x1, y1, slope, intercept)
            x2_proj, y2_proj = projection_point(x2, y2, slope, intercept)
            
            dist1 = distance(end_x, end_y, x1_proj, y1_proj)
            dist2 = distance(end_x, end_y, x2_proj, y2_proj)
            
            length = distance(x1_proj, y1_proj, x2_proj, y2_proj)
            
            if dist2 < dist1:
                return length
            else:
                return -length

        # while t < tf:
        #     k1 = dt * state_derivatives(globals.current_state[0:8], action)
        #     k2 = dt * state_derivatives(globals.current_state[0:8] + 1/2 * k1, action)
        #     k3 = dt * state_derivatives(globals.current_state[0:8] + 3/4 * k1, action)
        #     k4 = dt * state_derivatives(globals.current_state[0:8] + 2/9 * k1 + 1/3 * k2 + 4/9 * k3, action)

        #     state_2 = globals.current_state[0:8] + 7/24 * k1 + 1/4 * k2 + 1/3 * k3 + 1/8 * k4
        #     state_1 = globals.current_state[0:8] + 2/9 * k1 + 1/3 * k2 + 4/9 * k3

        #     error = np.abs(state_2 - state_1)

        #     if np.max(error) > tol:
        #         dt *= 0.9 * (tol / np.max(error))**0.25
        #     else:
        #         t += dt
        #         globals.current_state[0:8] = state_2
        #         self.state[0:8] = globals.current_state[0:8]
        #         dt *= 0.9 * (tol / np.max(error))**0.20

        #     if t + dt >tf:
        #         dt = tf - t

        while t < tf:
            y = globals.current_state[0:8]

            # Dormand-Prince coefficients (from MATLAB ode45)
            k1 = dt * state_derivatives(y, action)
            k2 = dt * state_derivatives(y + (1/5) * k1, action)
            k3 = dt * state_derivatives(y + (3/40) * k1 + (9/40) * k2, action)
            k4 = dt * state_derivatives(y + (44/45) * k1 - (56/15) * k2 + (32/9) * k3, action)
            k5 = dt * state_derivatives(y + (19372/6561) * k1 - (25360/2187) * k2 +
                                        (64448/6561) * k3 - (212/729) * k4, action)
            k6 = dt * state_derivatives(y + (9017/3168) * k1 - (355/33) * k2 +
                                        (46732/5247) * k3 + (49/176) * k4 -
                                        (5103/18656) * k5, action)

            # 5th-order accurate solution
            y_high = y + (35/384) * k1 + (500/1113) * k3 + (125/192) * k4 - \
                    (2187/6784) * k5 + (11/84) * k6

            # 4th-order embedded solution for error estimate
            k7 = dt * state_derivatives(y_high, action)
            y_low = y + (5179/57600) * k1 + (7571/16695) * k3 + (393/640) * k4 - \
                    (92097/339200) * k5 + (187/2100) * k6 + (1/40) * k7

            # Error estimation
            error = np.abs(y_high - y_low)

            if np.max(error) > tol:
                # Error too large, reduce timestep
                dt *= 0.9 * (tol / np.max(error))**0.2
            else:
                # Accept step
                t += dt
                globals.current_state[0:8] = y_high
                self.state[0:8] = y_high
                # Adjust step for next iteration
                dt *= 0.9 * (tol / np.max(error))**0.25

            # Prevent overshooting final time
            if t + dt > tf:
                dt = tf - t

        globals.current_time += tf
        globals.current_state[10] = projection_length(previous_state[0], previous_state[1], globals.current_state[0], globals.current_state[1])
        globals.current_state[8] = globals.current_state[8] + action[0] // (np.pi/180) * (np.pi/180)
        globals.current_state[9] = globals.current_state[9] + action[1] // (np.pi/180) * (np.pi/180)
        
        # globals.current_state[10] = 1
        # self.state[10] = globals.current_state[10]

        if globals.current_state[8] > np.pi/2:
            globals.current_state[8] = np.pi/2
            sail_error = True
        elif globals.current_state[8] < -np.pi/2:
            globals.current_state[8] = -np.pi/2
            sail_error = True

        if globals.current_state[9] > np.pi/6:
            globals.current_state[9] = np.pi/6
            # rudder_error = True
        elif globals.current_state[9] < -np.pi/6:
            globals.current_state[9] = -np.pi/6
            # rudder_error = True

        globals.total_energy = globals.total_energy + abs((action[1]) / (np.pi/6)) * 30 * 0.1 + abs((action[0]) / (np.pi/2)) * 90 * 0.4

        if distance(0,0,globals.current_state[0],globals.current_state[1]) >= 300 or globals.current_time >= globals.total_time or globals.total_energy >= 300:
            truncated = True
        else:
            truncated = False

        self.update_wind()
        reward = self.calculate_reward(previous_state,truncated)
        
        terminated = self.check_done()

        globals.state[0:11] = np.copy(globals.current_state)
        # globals.state[13:24] = np.copy(previous_state)

        if terminated or truncated:
            globals.path.append((self.state[0], self.state[1]))
        
        if self.render_mode == 'human':
            self.render()

        return globals.state, reward, terminated, truncated, {}
    
    def update_wind(self):          
            globals.w_change_v = 0.3 * globals.w_ini_v + 0.7 * globals.w_last_v + globals.var_v * np.random.randn()
            globals.w_change_d = 0.3 * globals.w_ini_d + 0.7 * globals.w_last_d + globals.var_d * np.random.randn()
            globals.w_last_v = globals.w_change_v
            globals.w_last_d = globals.w_change_d        

    def calculate_reward(self,laststate,truncated):
        
        # def projection_point(x, y, slope):
        #     # 计算点 (x, y) 在直线上的投影点
        #     if slope is None:
        #         return 0, y  # 垂直线的投影点
        #     x_proj = (x + slope * y) / (1 + slope ** 2)
        #     return x_proj
        
        # reward = 0
        # reward_scale = 0.2

        # #意外中止
        # if truncated:
        #     reward -= 500

        # #终点坐标
        # end_pos_x, end_pos_y = self.state[11:13]

        # #风向
        # wind_dir = self.state[13]

        # #分时坐标位置
        # pre_x,pre_y = laststate[0],laststate[1]
        # cur_x = np.copy(globals.current_state[0])
        # cur_y = np.copy(globals.current_state[1])

        # #中点检测
        # half_single = projection_point(cur_x, cur_y, globals.slope)/end_pos_x >= 1/2
        # #3/4检测
        # tf_single = cur_x/end_pos_x >= 3/4
        # #1/4检测
        # of_single = cur_x/end_pos_x >= 1/4
        
        #角度检测
        # ini_angle = math.atan2(end_pos_y, end_pos_x)
        # cur_angle = math.atan2(end_pos_y - cur_y, end_pos_x - cur_x)
        # pre_angle = math.atan2(end_pos_y - pre_y, end_pos_x - pre_x)

        # psi_angle = np.copy(globals.current_state[3])
        # # psi_pre_angle = np.copy(laststate[3])

        # psi_angle = (psi_angle + np.pi) % (2 * np.pi) - np.pi

        # # angle_ini_diff = psi_angle - ini_angle
        # # angle_ini_diff = (angle_ini_diff + np.pi) % (2 * np.pi) - np.pi

        # # angle_cur_diff = psi_angle - cur_angle
        # # angle_cur_diff = (angle_cur_diff + np.pi) % (2 * np.pi) - np.pi

        # # angle_pre_diff = psi_pre_angle - cur_angle
        # # angle_pre_diff = (angle_pre_diff + np.pi) % (2 * np.pi) - np.pi

        # # if half_single:
        # #     globals.mid_check += 1

        # # if tf_single:
        # #     globals.tf_check += 1
        
        # # if of_single:
        # #     globals.of_check += 1

        # # if globals.mid_check == 1:
        # #     reward += 25

        # # if globals.tf_check == 1:
        # #     reward += 50

        # # if globals.of_check == 1:
        # #     reward += 15

        # # #终点大奖励
        # # distance_to_target = np.sqrt((cur_x - end_pos_x)**2 + (cur_y - end_pos_y)**2)
        # # if distance_to_target <= 5.0:
        # #     reward += 200 + (300 - globals.total_energy)

        # # #速度修正
        # # if abs(angle_ini_diff) <= np.pi*5/12:
        # #     alpha_v = min(1/abs(np.cos(angle_ini_diff)),2)
        # # else:
        # #     alpha_v = 0.4

        # #禁航区设定
        # wind_diff_angle = (wind_dir - psi_angle + np.pi) % (2 * np.pi) - np.pi
        # if abs(wind_diff_angle) <= np.pi * 5 / 6:
        #     reward += 1
        # else:
        #     reward -= 5


        # pre_distance = np.sqrt((pre_x - end_pos_x)**2 + (pre_y - end_pos_y)**2)
        # cur_distance = np.sqrt((cur_x - end_pos_x)**2 + (cur_y - end_pos_y)**2)

        # removed_signal = cur_distance >= pre_distance
        
        #检测风场的方法
        #0329逆风运行时后半段不会主动靠近目标,修正——当有效位置超过3/4，但远离终点时，进行惩罚
        
        # if tf_single and removed_signal:
        #     reward -= 15
        
        # reward = reward + globals.current_state[10] + alpha_v * globals.current_state[4] - (cur_distance - pre_distance)



        # # #远离奖惩
        # # if removed_signal:
        # #     reward += (globals.current_state[4] - 1) * alpha_v + 5 * globals.current_state[10]
        # # else:
        # #     reward += globals.current_state[4] * alpha_v + 5 * globals.current_state[10]
        
        # #时间惩罚
        # reward -= 2

        # #速度奖励
        # if globals.current_state[4] > 1:
        #     reward += 0
        # else:
        #     reward -= 1

        # if globals.current_state[4] < abs(globals.current_state[5]):
        #     reward -= abs(globals.current_state[5])


        # #帆舵能源消耗约束       
        # if (globals.current_action[1]) != 0:
        #     reward -= abs( globals.current_action[1] / (np.pi/6)) * 30 * 0.1
        # else:
        #     reward -= 0

        # if (globals.current_action[0]) != 0:
        #     reward -= abs( globals.current_action[0] / (np.pi/2)) * 90 * 4 * 0.1
        # else:
        #     reward -= 0

        # if globals.current_state[8] == np.pi/2 and globals.current_action[0] > 0:
        #     reward -= 10

        # if globals.current_state[8] == -np.pi/2 and globals.current_action[0] < 0:
        #     reward -= 10

        # if globals.current_state[9] == np.pi/6 and globals.current_action[1] > 0:
        #     reward -= 5
        
        # if globals.current_state[9] == -np.pi/6 and globals.current_action[1] < 0:
        #     reward -= 5
        

        # reward = reward * reward_scale
         
        # return reward
    
        reward = 0
        reward_scale = 0.1
        reward_scale_sail = 0
        reward_scale_rudder = 0
        dis_diff = 0
        x_dis_diff = 0
        #终点坐标
        end_pos_x, end_pos_y = self.state[11:13]

        #风向
        wind_dir = self.state[13]
        psi_angle = np.copy(globals.current_state[3])
        # psi_pre_angle = np.copy(laststate[3])

        psi_angle = (psi_angle + np.pi) % (2 * np.pi) - np.pi
        #分时坐标位置
        pre_x,pre_y = laststate[0],laststate[1]
        cur_x = np.copy(globals.current_state[0])
        cur_y = np.copy(globals.current_state[1])

        #禁航区设定
        wind_diff_angle = (wind_dir - psi_angle + np.pi) % (2 * np.pi) - np.pi
        if abs(wind_diff_angle) <= np.pi * 5 / 6:
            reward += 0
        else:
            reward -= 5


        pre_distance = np.sqrt((pre_x - end_pos_x)**2 + (pre_y - end_pos_y)**2)
        cur_distance = np.sqrt((cur_x - end_pos_x)**2 + (cur_y - end_pos_y)**2)

        cur_x = np.copy(globals.current_state[0])
        cur_y = np.copy(globals.current_state[1])
        x_dis_diff = np.copy(globals.current_state[10])
        # x_dis_diff = np.copy(globals.current_state[11])
        u = np.copy(globals.current_state[4])
        v = np.copy(globals.current_state[5])
        reward += (2 * (pre_distance - cur_distance) + x_dis_diff) + (u / (1 + abs(v))) * 0.5

        sail = np.copy(globals.current_action[0])
        rudder = np.copy(globals.current_action[1])

        if abs(sail) <= np.pi/180:
            reward_scale_sail = reward_scale / 10
        else:
            reward -= abs(sail) // (np.pi/180) * (np.pi/180) * 0.5
        
        if abs(rudder) <= np.pi/180:
            reward_scale_rudder = reward_scale / 10
        else:
            reward -= abs(rudder) // (np.pi/180) * (np.pi/180) * 0.5
        
        reward = reward * (reward_scale + reward_scale_sail + reward_scale_rudder) 

        x = np.copy(globals.current_state[0])
        y = np.copy(globals.current_state[1])

        dis_to_target = math.sqrt((x - globals.state[11])**2 + (y - globals.state[12])**2)
        if dis_to_target <= 150 and globals.one_check == 0:
            reward += 2
            globals.one_check = 1
        if dis_to_target <= 100 and globals.two_check == 0:
            reward += 3
            globals.two_check = 1
        if dis_to_target <= 50 and globals.three_check == 0:
            reward += 5
            globals.three_check = 1

        if dis_to_target <= 5.0:
            reward += 20

        if truncated:
            reward -= 20 

        reward -= 0.2
         
        return reward



    def check_done(self):
        # 检查是否结束的逻辑
        x, y = self.state[0], self.state[1]

        distance_to_target = np.sqrt((x - self.state[11])**2 + (y - self.state[12])**2)
        reached_target = distance_to_target <= 5.0

        return reached_target

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            ):
        super().reset(seed=seed)

        globals.current_state = None
        globals.current_action = None
        globals.previous_state = None
        globals.state = None
        globals.previous_action = [0, 0]
        globals.current_time = 0

        globals.path = []

        # 风场状态
        globals.w0 = [5, 0]
        # globals.w0[1] = random.uniform(-np.pi, np.pi)
        # def generate_random_angle():
        #     if random.random() < 0.5:
        #         return random.uniform(-np.pi, -np.pi/3*2)
        #     else:
        #         return random.uniform(np.pi/3*2, np.pi)
        # globals.w0[1] = generate_random_angle()
        globals.w0[1] = np.pi * 6 / 6

        globals.var_v = 0.5
        globals.var_d = 1/18
        globals.w_change_v = 5
        globals.w_change_d = 0
        globals.w_last_v = globals.w0[0]
        globals.w_last_d = globals.w0[1]
        globals.w_ini_v = globals.w0[0]
        globals.w_ini_d = globals.w0[1]
        globals.one_check = 0
        globals.two_check = 0
        globals.three_check = 0
        globals.total_energy = 0

        # 初始化状态
        self.state = np.zeros(15,dtype=np.float32)



        #设定(x,y)到目标(0,0)为固定值200
        # theta = random.uniform(0, 2*np.pi)
        # self.state[0] = 200 * np.cos(theta)
        # self.state[1] = 200 * np.sin(theta)

        self.state[0] = 0
        self.state[13] = 5
        self.state[1] = 0
        self.state[14] = 0
        theta = random.uniform(0, 2*np.pi)
        # self.state[11] = 200 * np.cos(theta)
        # self.state[12] = 200 * np.sin(theta)
        self.state[11] = 200
        self.state[12] = 0

        if self.state[11] == 0:
            globals.slope = None
            globals.intercept = 0
        globals.slope = (self.state[12] - 0) / (self.state[11] - 0)
        globals.intercept = 0

        #设定船的朝向
        # self.state[3] = random.uniform(-np.pi, np.pi)
        # self.state[16] = self.state[3]
        self.state[3] = np.pi * 6 / 6
        globals.state = self.state

        return self.state, {}
    
    
    def render(self):

        global path, end_pos, current_state
        # 渲染环境
        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((800, 600))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((800, 600))

        # 绘制背景
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # 绘制起点和终点
        # pygame.draw.circle(self.surf, (0, 255, 0), self.start_pos, 10)  # 起点为绿色圆点
        pygame.draw.circle(self.surf, (255, 0, 0), self.state[11:13], 10)  # 终点为红色圆点

        # 绘制帆船
        boat_pos = (int(self.state[0]), int(self.state[1]))
        pygame.draw.circle(self.surf, (0, 0, 255), boat_pos, 5)  # 帆船为蓝色小圆点

        # 绘制运动路径
        if len(self.path) > 1:
            pygame.draw.lines(self.surf, (255, 0, 0), False, globals.path, 2)  # 路径为红色线条

        if self.render_mode == 'human':
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)  # 控制帧率为 60 FPS
        elif self.render_mode == 'rgb_array':
            return np.array(pygame.surfarray.pixels3d(self.surf))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()  