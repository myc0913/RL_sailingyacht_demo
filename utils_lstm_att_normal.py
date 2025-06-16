import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import globals
import math

# 添加Z-score标准化类
class ZScoreNormalizer:
    def __init__(self, state_dim, device):
        self.device = device
        self.count = 0
        self.mean = torch.zeros(state_dim, device=device)
        self.var = torch.ones(state_dim, device=device)
        self.epsilon = 1e-6
        
    def update(self, data):
        # data shape: (batch_size, seq_len, state_dim) 或 (seq_len, state_dim)
        if len(data.shape) == 3:
            data = data.view(-1, data.shape[-1])  # flatten to (batch*seq, state_dim)
        elif len(data.shape) == 2:
            pass  # already correct shape
        else:
            data = data.unsqueeze(0)  # single sample
            
        batch_count = data.shape[0]
        batch_mean = torch.mean(data, dim=0)
        batch_var = torch.var(data, dim=0, unbiased=False)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            # Online update
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / total_count
            self.var = (self.count * self.var + batch_count * batch_var + 
                       delta**2 * self.count * batch_count / total_count) / total_count
        
        self.count += batch_count
        
    def normalize(self, data):
        # Normalize data using current statistics
        return (data - self.mean) / (torch.sqrt(self.var) + self.epsilon)
        
    def denormalize(self, data):
        # Denormalize data
        return data * (torch.sqrt(self.var) + self.epsilon) + self.mean
        
    def state_dict(self):
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }
        
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.device)
        self.var = state_dict['var'].to(self.device)
        self.count = state_dict['count']

def build_net(layer_shape, hidden_activation, output_activation, dropout_prob=0.1):
    '''构建包含Dropout正则化的网络'''
    layers = []
    for j in range(len(layer_shape)-1):
        layers.append(nn.Linear(layer_shape[j], layer_shape[j+1]))
        # 对于隐藏层添加激活函数和Dropout（最后一层不加）
        if j < len(layer_shape)-2:
            layers.append(hidden_activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU, dropout_prob=0.1, bptt_len=10):
        super(Actor, self).__init__()
        
        # BPTT截断长度设为10
        self.bptt_len = bptt_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hid_shape[0]
        
        # 第一层：FC层 - 处理历史序列（状态+动作）
        self.fc_input = nn.Sequential(
            nn.Linear(state_dim + action_dim, hid_shape[0]),  # 状态+动作作为输入
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # 第二层：LSTM层 - 用于处理时序信息
        self.lstm = nn.LSTM(input_size=hid_shape[0], hidden_size=hid_shape[0], num_layers=1, batch_first=True)
        
        # 确保LSTM权重连续性
        self.lstm.flatten_parameters()
        
        # 添加Layer Normalization
        self.lstm_ln = nn.LayerNorm(hid_shape[0])
        
        # 当前状态预处理层
        self.current_state_fc = nn.Sequential(
            nn.Linear(state_dim, hid_shape[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # 第三层：拼接LSTM输出和预处理后的当前状态的FC层
        self.final_fc = nn.Sequential(
            nn.Linear(hid_shape[0] + hid_shape[0], hid_shape[-1]),  # 两个hidden_size维度拼接
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
            
        # 最终输出层
        self.mu_layer = nn.Linear(hid_shape[-1], action_dim)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -10
        
        # 为多环境维护LSTM隐藏状态
        self.hidden_states = {}

    def forward(self, state_sequence=None, action_sequence=None, current_state=None, deterministic=False, with_logprob=False, env_ids=None, 
                single_step_mode=False, prev_state=None, prev_action=None):
        '''
        两种模式：
        1. 序列模式 (single_step_mode=False): 用于训练，输入完整序列
           - state_sequence: (batch, seq_len, state_dim) - 历史状态序列
           - action_sequence: (batch, seq_len, action_dim) - 历史动作序列
           - current_state: (batch, state_dim) - 当前状态
           
        2. 单步模式 (single_step_mode=True): 用于环境采样，逐步更新
           - prev_state: (batch, state_dim) - 上一时刻状态
           - prev_action: (batch, action_dim) - 上一时刻动作
           - current_state: (batch, state_dim) - 当前状态
        '''
        device = current_state.device
        batch_size = current_state.shape[0]
        
        # 确保LSTM权重连续性
        self.lstm.flatten_parameters()
        
        if single_step_mode:
            # 单步模式：用于环境采样
            if prev_state is None or prev_action is None:
                raise ValueError("在单步模式下，必须提供prev_state和prev_action")
            
            # 拼接上一时刻的状态和动作
            prev_sa = torch.cat([prev_state, prev_action], dim=1)  # (batch, state_dim+action_dim)
            
            # FC层处理
            x = self.fc_input(prev_sa.unsqueeze(1))  # (batch, 1, hidden_size)
            
            # 管理LSTM隐藏状态
            if env_ids is not None:
                # 获取或初始化各环境的隐藏状态
                h_list, c_list = [], []
                for env_id in env_ids:
                    if env_id not in self.hidden_states:
                        # 初始化新环境的隐藏状态
                        h = torch.zeros(1, self.hidden_size, device=device)
                        c = torch.zeros(1, self.hidden_size, device=device)
                        self.hidden_states[env_id] = (h, c)
                    
                    h, c = self.hidden_states[env_id]
                    h_list.append(h)
                    c_list.append(c)
                
                # 拼接所有环境的隐藏状态
                h_0 = torch.cat(h_list, dim=0).unsqueeze(0)  # (1, batch, hidden_size)
                c_0 = torch.cat(c_list, dim=0).unsqueeze(0)  # (1, batch, hidden_size)
                
                # LSTM前向传播（单步）
                lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
                
                # 更新各环境的隐藏状态
                for i, env_id in enumerate(env_ids):
                    self.hidden_states[env_id] = (h_n[0, i:i+1, :], c_n[0, i:i+1, :])
            else:
                # 没有指定环境ID时，不保持状态
                lstm_out, _ = self.lstm(x)
            
            lstm_out = self.lstm_ln(lstm_out)
            lstm_final = lstm_out[:, 0, :]  # (batch, hidden_size)
            
        else:
            # 序列模式：用于训练
            if state_sequence is None or action_sequence is None:
                raise ValueError("在序列模式下，必须提供state_sequence和action_sequence")
            
            # 拼接历史状态和动作序列
            sa_sequence = torch.cat([state_sequence, action_sequence], dim=2)  # (batch, seq_len, state_dim+action_dim)
            
            # 第一层：FC层处理历史序列
            x = self.fc_input(sa_sequence)  # (batch, seq_len, hidden_size)
            
            # LSTM前向传播（序列模式不使用持久隐藏状态）
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_ln(lstm_out)
            
            # 取LSTM最后一个时间步的输出
            lstm_final = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 预处理当前状态
        current_state_processed = self.current_state_fc(current_state)  # (batch, hidden_size)
        
        # 拼接LSTM输出和预处理后的当前状态
        combined = torch.cat([lstm_final, current_state_processed], dim=1)  # (batch, hidden_size + hidden_size)
        
        # 最终FC层
        final_out = self.final_fc(combined)

        mu = self.mu_layer(final_out)
        log_std = self.log_std_layer(final_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        a = torch.tanh(u)
        if with_logprob:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (
                2 * (np.log(2) - u - F.softplus(-2 * u))
            ).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

    def reset_hidden_state(self, env_id=None):
        """重置LSTM隐藏状态"""
        if env_id is None:
            # 重置所有环境的隐藏状态
            self.hidden_states = {}
        else:
            # 重置特定环境的隐藏状态
            if env_id in self.hidden_states:
                del self.hidden_states[env_id]


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape_q1, hid_shape_q2, dropout_prob=0.1, bptt_len=10):
        super(Double_Q_Critic, self).__init__()
        
        # BPTT截断长度设为10
        self.bptt_len = bptt_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size_q1 = hid_shape_q1[0]
        self.hidden_size_q2 = hid_shape_q2[0]
        
        # Q_1 网络
        self.fc_input_q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hid_shape_q1[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.lstm_q1 = nn.LSTM(input_size=hid_shape_q1[0], hidden_size=hid_shape_q1[0], num_layers=1, batch_first=True)
        self.lstm_q1.flatten_parameters()
        self.lstm_ln_q1 = nn.LayerNorm(hid_shape_q1[0])
        
        # Q1当前状态动作预处理层
        self.current_sa_fc_q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hid_shape_q1[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        self.final_fc_q1 = nn.Sequential(
            nn.Linear(hid_shape_q1[0] + hid_shape_q1[0], hid_shape_q1[-1]),  # 两个hidden_size维度拼接
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.Q_1 = nn.Linear(hid_shape_q1[-1], 1)

        # Q_2 网络
        self.fc_input_q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hid_shape_q2[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.lstm_q2 = nn.LSTM(input_size=hid_shape_q2[0], hidden_size=hid_shape_q2[0], num_layers=1, batch_first=True)
        self.lstm_q2.flatten_parameters()
        self.lstm_ln_q2 = nn.LayerNorm(hid_shape_q2[0])
        
        # Q2当前状态动作预处理层
        self.current_sa_fc_q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hid_shape_q2[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        self.final_fc_q2 = nn.Sequential(
            nn.Linear(hid_shape_q2[0] + hid_shape_q2[0], hid_shape_q2[-1]),  # 两个hidden_size维度拼接
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.Q_2 = nn.Linear(hid_shape_q2[-1], 1)
        
        # 为多环境维护LSTM隐藏状态
        self.hidden_states_q1 = {}
        self.hidden_states_q2 = {}

    def forward(self, state_sequence=None, action_sequence=None, current_state=None, current_action=None, 
                env_ids=None, single_step_mode=False, prev_state=None, prev_action=None):
        '''
        两种模式：
        1. 序列模式 (single_step_mode=False): 用于训练，输入完整序列
           - state_sequence: (batch, seq_len, state_dim) - 历史状态序列
           - action_sequence: (batch, seq_len, action_dim) - 历史动作序列
           - current_state: (batch, state_dim) - 当前状态
           - current_action: (batch, action_dim) - 当前动作
           
        2. 单步模式 (single_step_mode=True): 用于环境采样，逐步更新
           - prev_state: (batch, state_dim) - 上一时刻状态
           - prev_action: (batch, action_dim) - 上一时刻动作
           - current_state: (batch, state_dim) - 当前状态
           - current_action: (batch, action_dim) - 当前动作
        '''
        device = current_state.device
        batch_size = current_state.shape[0]
        
        # 确保LSTM权重连续性
        self.lstm_q1.flatten_parameters()
        self.lstm_q2.flatten_parameters()
        
        if single_step_mode:
            # 单步模式：用于环境采样时的Critic评估
            if prev_state is None or prev_action is None:
                raise ValueError("在单步模式下，必须提供prev_state和prev_action")
            
            # 拼接上一时刻的状态和动作
            prev_sa = torch.cat([prev_state, prev_action], dim=1)  # (batch, state_dim+action_dim)
            
            # 当前状态和动作
            current_sa = torch.cat([current_state, current_action], dim=1)  # (batch, state_dim+action_dim)
            
            # Q1网络 - 单步模式
            x1 = self.fc_input_q1(prev_sa.unsqueeze(1))  # (batch, 1, hidden_size)
            
            # 管理Q1的LSTM隐藏状态
            if env_ids is not None:
                h_list_q1, c_list_q1 = [], []
                for env_id in env_ids:
                    if env_id not in self.hidden_states_q1:
                        h = torch.zeros(1, self.hidden_size_q1, device=device)
                        c = torch.zeros(1, self.hidden_size_q1, device=device)
                        self.hidden_states_q1[env_id] = (h, c)
                    
                    h, c = self.hidden_states_q1[env_id]
                    h_list_q1.append(h)
                    c_list_q1.append(c)
                
                h_0_q1 = torch.cat(h_list_q1, dim=0).unsqueeze(0)
                c_0_q1 = torch.cat(c_list_q1, dim=0).unsqueeze(0)
                
                lstm_out1, (h_n_q1, c_n_q1) = self.lstm_q1(x1, (h_0_q1, c_0_q1))
                
                # 更新隐藏状态
                for i, env_id in enumerate(env_ids):
                    self.hidden_states_q1[env_id] = (h_n_q1[0, i:i+1, :], c_n_q1[0, i:i+1, :])
            else:
                lstm_out1, _ = self.lstm_q1(x1)
            
            lstm_out1 = self.lstm_ln_q1(lstm_out1)
            lstm_final1 = lstm_out1[:, 0, :]  # (batch, hidden_size)
            
            # Q2网络 - 单步模式
            x2 = self.fc_input_q2(prev_sa.unsqueeze(1))  # (batch, 1, hidden_size)
            
            # 管理Q2的LSTM隐藏状态
            if env_ids is not None:
                h_list_q2, c_list_q2 = [], []
                for env_id in env_ids:
                    if env_id not in self.hidden_states_q2:
                        h = torch.zeros(1, self.hidden_size_q2, device=device)
                        c = torch.zeros(1, self.hidden_size_q2, device=device)
                        self.hidden_states_q2[env_id] = (h, c)
                    
                    h, c = self.hidden_states_q2[env_id]
                    h_list_q2.append(h)
                    c_list_q2.append(c)
                
                h_0_q2 = torch.cat(h_list_q2, dim=0).unsqueeze(0)
                c_0_q2 = torch.cat(c_list_q2, dim=0).unsqueeze(0)
                
                lstm_out2, (h_n_q2, c_n_q2) = self.lstm_q2(x2, (h_0_q2, c_0_q2))
                
                # 更新隐藏状态
                for i, env_id in enumerate(env_ids):
                    self.hidden_states_q2[env_id] = (h_n_q2[0, i:i+1, :], c_n_q2[0, i:i+1, :])
            else:
                lstm_out2, _ = self.lstm_q2(x2)
            
            lstm_out2 = self.lstm_ln_q2(lstm_out2)
            lstm_final2 = lstm_out2[:, 0, :]  # (batch, hidden_size)
            
        else:
            # 序列模式：用于训练
            if state_sequence is None or action_sequence is None:
                raise ValueError("在序列模式下，必须提供state_sequence和action_sequence")
            
            # 拼接历史状态和动作序列
            sa_sequence = torch.cat([state_sequence, action_sequence], dim=2)  # (batch, seq_len, state_dim+action_dim)
            
            # 当前状态和动作
            current_sa = torch.cat([current_state, current_action], dim=1)  # (batch, state_dim+action_dim)
            
            # Q1网络 - 序列模式
            x1 = self.fc_input_q1(sa_sequence)  # (batch, seq_len, hidden_dim)
            lstm_out1, _ = self.lstm_q1(x1)  # 训练时不保持隐藏状态
            lstm_out1 = self.lstm_ln_q1(lstm_out1)
            lstm_final1 = lstm_out1[:, -1, :]  # (batch, hidden_dim)
            
            # Q2网络 - 序列模式
            x2 = self.fc_input_q2(sa_sequence)  # (batch, seq_len, hidden_dim)
            lstm_out2, _ = self.lstm_q2(x2)  # 训练时不保持隐藏状态
            lstm_out2 = self.lstm_ln_q2(lstm_out2)
            lstm_final2 = lstm_out2[:, -1, :]  # (batch, hidden_dim)
            
            current_sa = torch.cat([current_state, current_action], dim=1)
        
        # 预处理当前状态动作（两种模式共用）
        current_sa_processed1 = self.current_sa_fc_q1(current_sa)  # (batch, hidden_dim)
        current_sa_processed2 = self.current_sa_fc_q2(current_sa)  # (batch, hidden_dim)
        
        # 拼接LSTM输出和预处理后的当前状态动作
        combined1 = torch.cat([lstm_final1, current_sa_processed1], dim=1)  # (batch, hidden_dim + hidden_dim)
        combined2 = torch.cat([lstm_final2, current_sa_processed2], dim=1)  # (batch, hidden_dim + hidden_dim)
        
        final_out1 = self.final_fc_q1(combined1)
        final_out2 = self.final_fc_q2(combined2)
        
        q1 = self.Q_1(final_out1)
        q2 = self.Q_2(final_out2)

        return q1, q2

    def reset_hidden_state(self, env_id=None):
        """重置LSTM隐藏状态"""
        if env_id is None:
            # 重置所有环境的隐藏状态
            self.hidden_states_q1 = {}
            self.hidden_states_q2 = {}
        else:
            # 重置特定环境的隐藏状态
            if env_id in self.hidden_states_q1:
                del self.hidden_states_q1[env_id]
            if env_id in self.hidden_states_q2:
                del self.hidden_states_q2[env_id]


def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8
    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10
    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r


def Action_adapter(a, max_action):
    # from [-1,1] to [-max,max]
    return a * max_action

def Action_adapter_reverse(act, max_action):
    # from [-max,max] to [-1,1]
    return act / max_action


def evaluate_policy(env, agent, max_action, turns=3, total_steps=None):
    total_scores = 0
    
    for j in range(turns):
        s, info = env.reset()  # s: (state_dim,)
        done = False
        
        # 为每次评估使用唯一的环境ID，并重置该环境的隐藏状态
        eval_env_id = f"eval_{j}"
        agent.actor.reset_hidden_state(eval_env_id)
        
        # 初始化前一步状态和动作
        prev_state = s.copy()
        prev_action = np.zeros(agent.action_dim)  # 初始前一动作为0
        
        episode_score = 0
        while not done:
            # 使用单步模式选择动作
            a = agent.select_action(prev_state, prev_action, s, deterministic=True, 
                                  total_steps=total_steps, env_id=eval_env_id)
            
            # 转换动作到环境范围
            act = Action_adapter(a, max_action)
            
            # 执行动作
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)
            
            episode_score += r
            
            # 更新前一步状态和动作
            prev_state = s.copy()
            prev_action = a.copy()
            s = s_next
        
        total_scores += episode_score
    
    return int(total_scores / turns)


def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')