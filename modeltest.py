from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC import SAC_countinuous
import gymnasium as gym
import os, shutil
import argparse
import torch
import globals
import time
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
import copy
import matplotlib.pyplot as plt

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=9000, help='which model to load')
parser.add_argument('--wind_angle', type=int, default=180, help='Wind direction angle in degrees (0-360)')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(9e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(9e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(30e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.1, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)

# 设置设备
opt.dvc = torch.device('cpu')  # 这里假设使用 CPU，如果需要使用 GPU，请修改为 'cuda'

# 定义环境名称
EnvName = ['sailboat-wind-v1']
BrifEnvName = ['wind_v1_2025_0329']

# 创建环境
env = gym.make(EnvName[opt.EnvIdex])
env.reset(seed=opt.seed)
opt.state_dim = env.observation_space.shape[0]
opt.action_dim = env.action_space.shape[0]
opt.max_action = env.action_space.high.astype(float)
opt.max_e_steps = 900

# 加载预训练模型
agent = SAC_countinuous(**vars(opt))

if not os.path.exists('model'): os.mkdir('model')
agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
# model_path = f'./model/{BrifEnvName[opt.EnvIdex]}_actor{opt.ModelIdex}.pth'
# agent.load(model_path)

# 运行环境并记录状态和动作参数
s, info = env.reset()

steps = []
actions = []
observations = []
wind_changes_v = []
wind_changes_d = []

done = False


step = 0
total_scores = 0

while not done:

    # 选择动作
    a = agent.select_action(s, deterministic=True)
    act = Action_adapter(a, opt.max_action)
    # 执行动作
    # act = [0,0]
    s_next, r, dw, tr, info = env.step(act)
    
    # 收集数据
    actions.append(act)
    observations.append(copy.deepcopy(s_next))
    wind_changes_v.append(globals.w_change_v)
    wind_changes_d.append(globals.w_change_d)

    # 输出当前步的信息
    print(f"Step: {step}")
    print(f"Action: {act}")
    print(f"Observation: {s_next}")
    print(f"Reward: {r}")
    print(f"Terminated: {dw}")
    print(f"Truncated: {tr}")
    print(f"Info: {info}")
    print("-" * 50)

    # 判断是否结束
    done = (dw or tr)
    # 记录状态和动作参数
    print(s, act)
    s = s_next
    step += 1
    total_scores += r
    print(f"step:{step}")
    print(f"all_reward:{total_scores}")

# 将数据保存到桌面
desktop_path = 'C:\\Users\\24809\\Desktop\\'
# 将 observations 转换为 NumPy 数组
observations_array = np.array(observations)

# 创建带风向角度标识的文件名后缀
file_suffix = f"_model_wind{opt.wind_angle}deg"

# 保存数据
np.savetxt(desktop_path + f'actions{file_suffix}.txt', np.array(actions), delimiter=',', header='Actions', comments='')
np.savetxt(desktop_path + f'observations{file_suffix}.txt', np.array(observations), delimiter=',', header=','.join([f'Observation {i}' for i in range(observations_array.shape[1])]), comments='')
np.savetxt(desktop_path + f'wind_changes{file_suffix}.txt', np.column_stack((wind_changes_v, wind_changes_d)), delimiter=',', header='Wind Change V,Wind Change D', comments='')

# 保存实验配置信息
config_info = f"""Wind Angle: {opt.wind_angle} degrees
Model Index: {opt.ModelIdex}
Total Steps: {step}
Total Reward: {total_scores}
Seed: {opt.seed}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(desktop_path + f'experiment_config{file_suffix}.txt', 'w') as f:
    f.write(config_info)

x = observations_array[:, 0]
y = observations_array[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.title(f'Trajectory of Observations - Wind {opt.wind_angle}°')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(-500, 500)  # 设置 x 轴范围
plt.ylim(-200, 200)  # 设置 y 轴范围
plt.grid(True)
plt.savefig(desktop_path + f'trajectory{file_suffix}.png', dpi=300, bbox_inches='tight')
plt.show()

env.close()