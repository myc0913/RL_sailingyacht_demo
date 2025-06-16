from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from SAC import SAC_countinuous
from datetime import datetime

import gymnasium as gym
import os
import argparse
import torch
import globals
import numpy as np
from tqdm import tqdm
import time
import json

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--ModelIdex', type=int, default=9000, help='which model to load')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_tests', type=int, default=100, help='Number of test episodes')
parser.add_argument('--max_e_steps', type=int, default=500, help='Maximum steps per episode')

# SAC相关参数
parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.1, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)

# 定义环境名称
EnvName = ['sailboat-wind-v1']
BrifEnvName = ['wind_v1_2025_0329']


def run_single_test(agent, env, opt, test_id):
    """运行单次测试"""
    # 重置环境
    s, info = env.reset(seed=opt.seed + test_id)
    
    step = 0
    total_reward = 0
    total_distance = 0  # 初始化总航行距离
    done = False
    
    # 记录初始位置
    prev_x, prev_y = s[0], s[1]
    
    while not done and step < opt.max_e_steps:
        # 选择动作
        a = agent.select_action(s, deterministic=True)
        
        # 转换动作到环境范围
        act = Action_adapter(a, opt.max_action)
        
        # 执行动作
        s_next, r, dw, tr, info = env.step(act)
        
        # 计算当前步的航行距离
        current_x, current_y = s_next[0], s_next[1]
        step_distance = np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
        total_distance += step_distance
        
        # 更新状态和奖励
        done = (dw or tr)
        s = s_next
        step += 1
        total_reward += r
        
        # 更新位置记录
        prev_x, prev_y = current_x, current_y
    
    # 判断任务是否完成（根据您的具体任务定义）
    task_completed = step >= opt.max_e_steps or dw  # 可根据实际需求调整
    
    return {
        'episode': test_id + 1,
        'steps': step,
        'total_reward': total_reward,
        'total_distance': total_distance,  # 添加总航行距离
        'task_completed': task_completed,
        'terminated': dw,
        'truncated': tr
    }

def main():
    print("开始大规模测试...")
    print(f"测试回合数: {opt.num_tests}")
    print(f"每回合最大步数: {opt.max_e_steps}")
    print("-" * 50)
    
    # 创建环境
    env = gym.make(EnvName[opt.EnvIdex])
    env.reset(seed=opt.seed)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = env.action_space.high.astype(float)
    
    # 加载预训练模型
    agent = SAC_countinuous(**vars(opt))
    
    # 检查模型文件是否存在
    if not os.path.exists('model'): 
        print("模型文件夹不存在！")
        return
    
    # 注意：这里不需要标准化文件，只需要actor和critic模型
    model_files = [
        f'./model/{BrifEnvName[opt.EnvIdex]}_actor{opt.ModelIdex}.pth',
        f'./model/{BrifEnvName[opt.EnvIdex]}_q_critic{opt.ModelIdex}.pth'
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"模型文件不存在: {model_file}")
            return
    
    try:
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
        print(f"成功加载模型: {BrifEnvName[opt.EnvIdex]}_{opt.ModelIdex}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 存储测试结果
    results = []
    
    # 运行测试
    start_time = time.time()
    
    for test_id in tqdm(range(opt.num_tests), desc="测试进度"):
        result = run_single_test(agent, env, opt, test_id)
        results.append(result)
    
    end_time = time.time()
    
    # 筛选成功的案例（成功标准：terminated为True）
    successful_results = [r for r in results if r['terminated']]
    failed_results = [r for r in results if not r['terminated']]
    
    # 计算统计信息（仅基于成功案例）
    total_episodes = len(results)
    successful_episodes = len(successful_results)
    failed_episodes = len(failed_results)
    
    if successful_episodes > 0:
        # 仅对成功案例计算统计指标
        successful_rewards = [r['total_reward'] for r in successful_results]
        successful_steps = [r['steps'] for r in successful_results]
        successful_distances = [r['total_distance'] for r in successful_results]  # 添加距离统计
        successful_completed = [r['task_completed'] for r in successful_results]
        successful_terminated = [r['terminated'] for r in successful_results]
        
        # 计算性能指标
        stats = {
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'failed_episodes': failed_episodes,
            'success_rate': (successful_episodes / total_episodes) * 100,
            'task_completion_rate': np.mean(successful_completed) * 100,
            'termination_rate': np.mean(successful_terminated) * 100,
            'average_total_reward': np.mean(successful_rewards),
            'std_total_reward': np.std(successful_rewards),
            'min_total_reward': np.min(successful_rewards),
            'max_total_reward': np.max(successful_rewards),
            'average_steps': np.mean(successful_steps),
            'std_steps': np.std(successful_steps),
            'min_steps': np.min(successful_steps),
            'max_steps': np.max(successful_steps),
            'average_total_distance': np.mean(successful_distances),  # 添加距离统计
            'std_total_distance': np.std(successful_distances),
            'min_total_distance': np.min(successful_distances),
            'max_total_distance': np.max(successful_distances),
            'total_test_time': end_time - start_time,
            'average_time_per_episode': (end_time - start_time) / total_episodes
        }
    else:
        # 如果没有成功案例，创建空统计
        stats = {
            'total_episodes': total_episodes,
            'successful_episodes': 0,
            'failed_episodes': failed_episodes,
            'success_rate': 0.0,
            'task_completion_rate': 0.0,
            'termination_rate': 0.0,
            'average_total_reward': 0.0,
            'std_total_reward': 0.0,
            'min_total_reward': 0.0,
            'max_total_reward': 0.0,
            'average_steps': 0.0,
            'std_steps': 0.0,
            'min_steps': 0,
            'max_steps': 0,
            'average_total_distance': 0.0,  # 添加距离统计
            'std_total_distance': 0.0,
            'min_total_distance': 0.0,
            'max_total_distance': 0.0,
            'total_test_time': end_time - start_time,
            'average_time_per_episode': (end_time - start_time) / total_episodes
        }
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果统计（仅统计成功案例）")
    print("="*60)
    print(f"总测试回合数: {stats['total_episodes']}")
    print(f"成功回合数: {stats['successful_episodes']}")
    print(f"失败回合数: {stats['failed_episodes']}")
    print(f"成功率: {stats['success_rate']:.2f}%")
    
    if successful_episodes > 0:
        print(f"任务完成率 (成功案例中): {stats['task_completion_rate']:.2f}%")
        print(f"正常终止率 (成功案例中): {stats['termination_rate']:.2f}%")
        print(f"平均总奖励 (成功案例): {stats['average_total_reward']:.2f} ± {stats['std_total_reward']:.2f}")
        print(f"奖励范围 (成功案例): [{stats['min_total_reward']:.2f}, {stats['max_total_reward']:.2f}]")
        print(f"平均步数 (成功案例): {stats['average_steps']:.2f} ± {stats['std_steps']:.2f}")
        print(f"步数范围 (成功案例): [{stats['min_steps']:.0f}, {stats['max_steps']:.0f}]")
        print(f"平均总航行距离 (成功案例): {stats['average_total_distance']:.2f} ± {stats['std_total_distance']:.2f}")
        print(f"航行距离范围 (成功案例): [{stats['min_total_distance']:.2f}, {stats['max_total_distance']:.2f}]")
    else:
        print("注意: 所有测试案例均失败，无法计算成功案例的统计指标")
    
    print(f"总测试时间: {stats['total_test_time']:.2f} 秒")
    
    # 保存详细结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存统计信息
    desktop_path = 'C:\\Users\\24809\\Desktop\\'
    stats_filename = f'{desktop_path}batch_test_sac_stats_{timestamp}.json'
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计结果已保存至: {stats_filename}")
    
    # 保存简要结果摘要
    summary_filename = f'{desktop_path}batch_test_sac_summary_{timestamp}.txt'
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("SAC模型大规模测试结果摘要\n")
        f.write("="*40 + "\n")
        f.write(f"模型: {BrifEnvName[opt.EnvIdex]}_{opt.ModelIdex}\n")
        f.write(f"环境: {EnvName[opt.EnvIdex]}\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总测试回合数: {stats['total_episodes']}\n")
        f.write(f"成功率: {stats['success_rate']:.2f}%\n")
        f.write(f"任务完成率: {stats['task_completion_rate']:.2f}%\n")
        f.write(f"平均总奖励: {stats['average_total_reward']:.2f}\n")
        f.write(f"奖励标准差: {stats['std_total_reward']:.2f}\n")
        f.write(f"平均步数: {stats['average_steps']:.2f}\n")
        f.write(f"平均航行距离: {stats['average_total_distance']:.2f}\n")
        f.write(f"距离标准差: {stats['std_total_distance']:.2f}\n")
        f.write(f"总测试时间: {stats['total_test_time']:.2f} 秒\n")
    print(f"测试摘要已保存至: {summary_filename}")
    
    # 关闭环境
    env.close()
    print("测试完成，环境已关闭")

if __name__ == "__main__":
    main()