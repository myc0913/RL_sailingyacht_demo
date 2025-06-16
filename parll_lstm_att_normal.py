import os, shutil
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils_lstm_att_normal import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC_lstm_att_normal import SAC_lstm_countinuous
import gymnasium as gym
import argparse
import torch
import globals
import time
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import deque

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=0, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(9e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(9e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e5), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=8, help='Training Frequency, in steps')

parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.2, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--num_envs', type=int, default=20, help='Number of parallel environments')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def make_env(env_id, seed):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init

def main():
    EnvName = ['sailboat-s14-v1']
    BrifEnvName = ['saclstm_s14_0602_lstm1_att']

    # Build Env
    env_fns = [make_env(EnvName[opt.EnvIdex], opt.seed + i) for i in range(opt.num_envs)]
    env = SubprocVecEnv(env_fns)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = env.action_space.high.astype(float)

    opt.max_e_steps = 500
    seq_len = 10  # 序列长度
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a_s:{opt.max_action[0]}  max_a_r:{opt.max_action[1]} min_a_s:{env.action_space.low[0],env.action_space.low[1]}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_lstm_countinuous(**vars(opt)) # 传入所有参数

    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    # 初始化状态序列缓存 - 为每个环境维护独立的状态和动作序列
    s_raw = env.reset()  # s_raw: (num_envs, state_dim)
    
    # 为每个环境初始化状态和动作序列缓存（用于训练时的序列输入）
    state_buffers = [deque(maxlen=seq_len) for _ in range(opt.num_envs)]
    action_buffers = [deque(maxlen=seq_len) for _ in range(opt.num_envs)]
    
    # 为每个环境维护前一步的状态和动作（用于采样时的单步输入）
    prev_states = [None for _ in range(opt.num_envs)]
    prev_actions = [np.zeros(opt.action_dim) for _ in range(opt.num_envs)]  # 初始前一动作为0
    
    # 用初始状态填充缓存
    for i in range(opt.num_envs):
        for _ in range(seq_len):
            state_buffers[i].append(s_raw[i].copy())
            action_buffers[i].append(np.zeros(opt.action_dim))  # 初始动作为0
        prev_states[i] = s_raw[i].copy()  # 初始化前一状态
    
    current_states = s_raw  # 当前状态

    if opt.render:
        # 评估时也需要创建状态序列
        eval_state_buffer = deque(maxlen=seq_len)
        while True:
            score = evaluate_policy(eval_env, agent, opt.max_action, turns=3)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        start_time = time.time()
        
        # 添加loss和alpha数据记录的变量
        training_log_interval = 20  # 每20步记录一次训练loss
        
        with tqdm(total=opt.Max_train_steps, desc='Training') as pbar:
            while total_steps < opt.Max_train_steps:
                done_flags = [False] * opt.num_envs

                '''Interact & train'''
                while not all(done_flags):
                    actions = []
                    
                    # 为每个环境选择动作
                    for i in range(opt.num_envs):
                        # 在训练初期可选随机动作，否则使用 agent.select_action
                        if total_steps < (20 * opt.max_e_steps):
                            # 针对帆船控制的特殊随机策略
                            random_strategy = np.random.random()
                            
                            if random_strategy < 0.3:  # 30%概率：保持稳定状态（0动作）
                                a = np.zeros(opt.action_dim)
                            elif random_strategy < 0.5:  # 20%概率：小幅调整（模拟微调）
                                # 小幅度调整，适合帆船的渐进式控制
                                a = np.random.uniform(-0.2, 0.2, opt.action_dim)
                            elif random_strategy < 0.65:  # 15%概率：单维度调整
                                # 只调整帆或只调整舵，避免同时大幅调整
                                a = np.zeros(opt.action_dim)
                                if opt.action_dim >= 2:
                                    # 假设第一个维度是帆，第二个是舵
                                    if np.random.random() < 0.5:
                                        a[0] = np.random.uniform(-0.5, 0.5)  # 调整帆
                                    else:
                                        a[1] = np.random.uniform(-0.5, 0.5)  # 调整舵
                                else:
                                    random_dim = np.random.randint(0, opt.action_dim)
                                    a[random_dim] = np.random.uniform(-0.5, 0.5)
                            else:  # 35%概率：正常随机动作
                                act = env.action_space.sample()
                                a = Action_adapter_reverse(act, opt.max_action)
                        else:
                            # 使用单步模式：前一状态、前一动作和当前状态选择动作
                            a = agent.select_action(prev_states[i], prev_actions[i], current_states[i], 
                                                  deterministic=False, total_steps=total_steps, env_id=i)
                        actions.append(a)
                    
                    # 将连续动作转换至环境所需范围
                    act_list = [Action_adapter(a, opt.max_action) for a in actions]
                    s_next_raw, r, done_flags, info = env.step(act_list)  # s_next_raw: (num_envs, state_dim)
                    
                    # 对每个环境存入ReplayBuffer并更新缓存
                    for i in range(opt.num_envs):
                        # 获取当前环境的序列数据（用于训练）
                        curr_state_seq = np.array(list(state_buffers[i]))  # (seq_len, state_dim)
                        curr_action_seq = np.array(list(action_buffers[i]))  # (seq_len, action_dim)
                        
                        # 更新动作缓存（添加当前选择的动作）
                        action_buffers[i].append(actions[i].copy())
                        # 更新状态缓存（添加下一个状态）
                        state_buffers[i].append(s_next_raw[i].copy())
                        
                        # 构建下一个状态的序列
                        next_state_seq = np.array(list(state_buffers[i]))  # (seq_len, state_dim)
                        next_action_seq = np.array(list(action_buffers[i]))  # 使用更新后的动作缓存

                        # 存储到经验池（训练时使用序列）
                        agent.replay_buffer.add(curr_state_seq, curr_action_seq, current_states[i], 
                                              actions[i], r[i], next_state_seq, next_action_seq, s_next_raw[i], done_flags[i])
                        
                        # 更新前一步状态和动作（用于下次采样）
                        prev_states[i] = current_states[i].copy()
                        prev_actions[i] = actions[i].copy()
                        
                        # 如果episode结束，重置该环境的相关状态
                        if done_flags[i]:
                            agent.actor.reset_hidden_state(env_id=i)
                            agent.q_critic.reset_hidden_state(env_id=i)
                            agent.q_critic_target.reset_hidden_state(env_id=i)
                            
                            # 重置状态和动作缓存
                            state_buffers[i].clear()
                            action_buffers[i].clear()
                            # 用新的初始状态填充缓存
                            for _ in range(seq_len):
                                state_buffers[i].append(s_next_raw[i].copy())
                                action_buffers[i].append(np.zeros(opt.action_dim))
                        
                            # 重置前一步状态和动作
                            prev_states[i] = s_next_raw[i].copy()
                            prev_actions[i] = np.zeros(opt.action_dim)

                    # 更新当前状态
                    current_states = s_next_raw

                    total_steps += opt.num_envs
                    pbar.update(opt.num_envs)

                    '''train after collecting samples from all environments'''
                    if (total_steps >= 20 * opt.max_e_steps) and (agent.replay_buffer.size >= opt.batch_size):
                        # 每20步进行一次训练序列：4次critic更新 + 1次actor和alpha更新
                        if total_steps % 20 == 0:  # 每20步环境采样进行一次训练序列
                            # 计算训练进度，用于更新beta值
                            progress = min(total_steps / 2000000, 1.0)
                            # 更新 beta 值
                            agent.replay_buffer.update_beta(progress)
                            
                            # 执行4次critic更新
                            for update_idx in range(opt.update_every):
                                train_info = agent.train()
                                
                                # 只在最后一次更新时记录日志（避免过多的记录）
                                if opt.write and update_idx == 3:  # 只在第4次更新时记录
                                    writer.add_scalar('Train/Q_Loss', train_info['q_loss'], global_step=total_steps)
                                    
                                    # 预热阶段特殊处理
                                    if train_info.get('warmup', False):
                                        writer.add_scalar('Train/Warmup_Progress', train_info['warmup_progress'], global_step=total_steps)
                                        writer.add_scalar('Train/Actor_Loss', 0.0, global_step=total_steps)
                                        writer.add_scalar('Train/Alpha_Loss', 0.0, global_step=total_steps)
                                    else:
                                        # 预热后：记录实际的loss值
                                        writer.add_scalar('Train/Actor_Loss', train_info['a_loss'], global_step=total_steps)
                                        writer.add_scalar('Train/Alpha_Loss', train_info['alpha_loss'], global_step=total_steps)
                                        
                                        # 记录策略更新状态
                                        writer.add_scalar('Train/Policy_Updated', float(train_info.get('policy_updated', False)), global_step=total_steps)
                                        writer.add_scalar('Train/Critic_Update_Count', train_info.get('critic_update_count', 0), global_step=total_steps)
                                    
                                    # 记录当前alpha值
                                    writer.add_scalar('Train/Alpha_Value', train_info['alpha'], global_step=total_steps)
                                    
                                    # 记录replay buffer beta值
                                    writer.add_scalar('Train/Buffer_Beta', agent.replay_buffer.beta, global_step=total_steps)
                                    
                                    # 记录buffer大小
                                    writer.add_scalar('Train/Buffer_Size', agent.replay_buffer.size, global_step=total_steps)
                            
                            # 强制清理GPU缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    '''record & log'''
                    if (total_steps >= 20 * opt.max_e_steps) and (total_steps % opt.eval_interval) == 0:
                        ep_r = evaluate_policy(eval_env, agent, opt.max_action, turns=3)
                        if opt.write: writer.add_scalar('Eval/Episode_Reward', ep_r, global_step=total_steps)
                        
                        elapsed_time = time.time() - start_time
                        
                        # 添加详细的训练状态打印，包括预热信息
                        current_alpha = agent.alpha.item() if hasattr(agent.alpha, 'item') else agent.alpha
                        recent_q_loss = agent.q_loss_history[-1] if len(agent.q_loss_history) > 0 else 0.0
                        recent_a_loss = agent.a_loss_history[-1] if len(agent.a_loss_history) > 0 else 0.0
                        recent_alpha_loss = agent.alpha_loss_history[-1] if len(agent.alpha_loss_history) > 0 else 0.0
                        
                        # 检查是否处于预热阶段
                        warmup_status = ""
                        if hasattr(agent, 'training_step') and agent.training_step <= agent.critic_warmup_steps:
                            warmup_progress = agent.training_step / agent.critic_warmup_steps * 100
                            warmup_status = f", Warmup: {warmup_progress:.1f}%"
                        
                        print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, '
                              f'Episode Reward:{ep_r}, Alpha:{current_alpha:.4f}, '
                              f'Q_Loss:{recent_q_loss:.4f}, A_Loss:{recent_a_loss:.4f}, '
                              f'Alpha_Loss:{recent_alpha_loss:.4f}, Elapsed Time:{elapsed_time:.2f}s{warmup_status}')

                    '''save model'''
                    if total_steps % opt.save_interval == 0:
                        agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
                        
                        # 保存训练统计信息
                        if opt.write:
                            training_stats = {
                                'q_loss_history': agent.q_loss_history[-1000:],  # 保存最近1000个loss
                                'a_loss_history': agent.a_loss_history[-1000:],
                                'alpha_loss_history': agent.alpha_loss_history[-1000:],
                                'total_steps': total_steps,
                                'current_alpha': agent.alpha.item() if hasattr(agent.alpha, 'item') else agent.alpha
                            }
                            torch.save(training_stats, f"./model/{BrifEnvName[opt.EnvIdex]}_training_stats_{int(total_steps/1000)}.pth")
                        
                if total_steps >= opt.Max_train_steps:
                    break
        
        # 训练完成后保存最终的训练统计信息
        if opt.write:
            final_stats = {
                'q_loss_history': agent.q_loss_history,
                'a_loss_history': agent.a_loss_history,  
                'alpha_loss_history': agent.alpha_loss_history,
                'total_steps': total_steps,
                'final_alpha': agent.alpha.item() if hasattr(agent.alpha, 'item') else agent.alpha
            }
            torch.save(final_stats, f"./model/{BrifEnvName[opt.EnvIdex]}_final_training_stats.pth")
            writer.close()
            
        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
