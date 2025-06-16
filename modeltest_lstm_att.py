from utils_lstm_att_normal import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC_lstm_att_normal import SAC_lstm_countinuous
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
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=5400, help='which model to load')
parser.add_argument('--wind_angle', type=int, default=180, help='Wind direction angle in degrees (0-360)')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(9e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(9e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e5), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.1, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)

# 定义环境名称
EnvName = ['sailboat-s14-v1']
BrifEnvName = ['saclstm_s14_0604_lstm1_att']  # 修改为正确的模型名称

# 创建环境
env = gym.make(EnvName[opt.EnvIdex])
env.reset(seed=opt.seed)
opt.state_dim = env.observation_space.shape[0]
opt.action_dim = env.action_space.shape[0]
opt.max_action = env.action_space.high.astype(float)
opt.max_e_steps = 900

# 加载预训练模型
agent = SAC_lstm_countinuous(**vars(opt))

if not os.path.exists('model'): 
    print("模型文件夹不存在！")
    exit()

# 检查模型文件是否存在
model_files = [
    f'./model/{BrifEnvName[opt.EnvIdex]}_actor{opt.ModelIdex}.pth',
    f'./model/{BrifEnvName[opt.EnvIdex]}_q_critic{opt.ModelIdex}.pth',
    f'./model/{BrifEnvName[opt.EnvIdex]}_normalizer{opt.ModelIdex}.pth'
]

for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"模型文件不存在: {model_file}")
        print("可用的模型文件:")
        for file in os.listdir('./model/'):
            if file.startswith(BrifEnvName[opt.EnvIdex]):
                print(f"  {file}")
        exit()

try:
    agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
    print(f"成功加载模型: {BrifEnvName[opt.EnvIdex]}_{opt.ModelIdex}")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# 环境初始化
s, info = env.reset()

# 初始化前一步的状态和动作（用于LSTM输入）
prev_state = s.copy()
prev_action = np.zeros(opt.action_dim)  # 初始前一动作为0

# 为测试环境分配一个唯一的环境ID，并重置隐藏状态
test_env_id = "test_env"
agent.actor.reset_hidden_state(test_env_id)

# 存储数据的列表
steps = []
actions = []
observations = []
wind_changes_v = []
wind_changes_d = []
rewards = []

step = 0
total_scores = 0
done = False

print("开始测试模型...")
print(f"初始状态: {s}")
print("-" * 50)

while not done and step < opt.max_e_steps:
    # 使用修正后的select_action方法
    a = agent.select_action(prev_state, prev_action, s, deterministic=True, 
                           total_steps=None, env_id=test_env_id)
    
    # 转换动作到环境范围
    act = Action_adapter(a, opt.max_action)
    
    # 执行动作，获得下一时刻的状态
    s_next, r, dw, tr, info = env.step(act)
    
    # 收集数据
    steps.append(step)
    actions.append(act.copy())
    observations.append(s_next.copy())
    rewards.append(r)
    
    # 收集风变化数据（如果全局变量存在）
    try:
        wind_changes_v.append(globals.w_change_v)
        wind_changes_d.append(globals.w_change_d)
    except AttributeError:
        # 如果全局变量不存在，使用默认值
        wind_changes_v.append(0.0)
        wind_changes_d.append(0.0)

    # 输出当前步信息
    print(f"Step: {step}")
    print(f"Previous State: {prev_state}")
    print(f"Previous Action: {prev_action}")
    print(f"Current State: {s}")
    print(f"Selected Action: {a}")
    print(f"Environment Action: {act}")
    print(f"Next State: {s_next}")
    print(f"Reward: {r}")
    print(f"Terminated: {dw}")
    print(f"Truncated: {tr}")
    print(f"Info: {info}")
    print("-" * 50)
    
    # 检查是否结束
    done = (dw or tr)
    
    # 更新前一步状态和动作
    prev_state = s.copy()
    prev_action = a.copy()
    s = s_next
    
    step += 1
    total_scores += r
    
    print(f"累计步数: {step}, 累计奖励: {total_scores:.2f}")

print(f"\n测试完成!")
print(f"总步数: {step}")
print(f"总奖励: {total_scores:.2f}")
print(f"平均奖励: {total_scores/step:.2f}")

# 数据保存部分
desktop_path = 'C:\\Users\\24809\\Desktop\\'

# 转换为numpy数组
observations_array = np.array(observations)
actions_array = np.array(actions)
steps_array = np.array(steps)
rewards_array = np.array(rewards)

# 创建带风向角度标识的文件名后缀
file_suffix = f"_model_lstm_att_wind{opt.wind_angle}deg"

# 保存数据
try:
    np.savetxt(desktop_path + f'actions{file_suffix}.txt', actions_array, delimiter=',', 
               header='Action_Sail,Action_Rudder', comments='')
    print("动作数据已保存")
    
    np.savetxt(desktop_path + f'observations{file_suffix}.txt', observations_array, delimiter=',', 
               header=','.join([f'Observation_{i}' for i in range(observations_array.shape[1])]), comments='')
    print("观测数据已保存")
    
    np.savetxt(desktop_path + f'rewards{file_suffix}.txt', rewards_array, delimiter=',', 
               header='Reward', comments='')
    print("奖励数据已保存")
    
    if len(wind_changes_v) > 0:
        np.savetxt(desktop_path + f'wind_changes{file_suffix}.txt', 
                   np.column_stack((wind_changes_v, wind_changes_d)), delimiter=',', 
                   header='Wind_Change_V,Wind_Change_D', comments='')
        print("风变化数据已保存")
    
except Exception as e:
    print(f"保存数据时出错: {e}")

# 绘图部分
try:
    if observations_array.shape[1] >= 2:
        x = observations_array[:, 0]
        y = observations_array[:, 1]
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 轨迹图
        plt.subplot(2, 2, 1)
        plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=3)
        plt.title('Trajectory of Agent')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        
        # 子图2: 奖励曲线
        plt.subplot(2, 2, 2)
        plt.plot(steps_array, rewards_array, color='r')
        plt.title('Reward over Time')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # 子图3: 动作曲线
        plt.subplot(2, 2, 3)
        plt.plot(steps_array, actions_array[:, 0], label='Sail Action', color='g')
        if actions_array.shape[1] > 1:
            plt.plot(steps_array, actions_array[:, 1], label='Rudder Action', color='orange')
        plt.title('Actions over Time')
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.legend()
        plt.grid(True)
        
        # 子图4: 累计奖励
        plt.subplot(2, 2, 4)
        cumulative_rewards = np.cumsum(rewards_array)
        plt.plot(steps_array, cumulative_rewards, color='purple')
        plt.title('Cumulative Reward')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(desktop_path + f'model_test_results{file_suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("图表已保存并显示")
    else:
        print("观测数据维度不足，无法绘制轨迹图")
        
except Exception as e:
    print(f"绘图时出错: {e}")

# 关闭环境
env.close()
print("测试完成，环境已关闭")