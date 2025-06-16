from utils_lstm_att_normal import Actor, Double_Q_Critic, ZScoreNormalizer
import torch.nn.functional as F
import numpy as np
import torch
import copy
import globals


class SAC_lstm_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.seq_len = 10  # 序列长度
        
        # 添加预热参数
        self.critic_warmup_steps = 0 # critic预热步数
        self.training_step = 0  # 训练步数计数器
        
        # 添加延迟更新参数
        self.policy_update_freq = 4  # 策略更新频率，critic更新2次actor更新1次
        self.critic_update_count = 0  # critic更新次数计数器
        
        # 添加梯度裁剪参数
        self.max_grad_norm = 1
        
        # 添加loss监控
        self.q_loss_history = []
        self.a_loss_history = []
        self.alpha_loss_history = []

        # 初始化Z-score标准化器
        self.state_normalizer = ZScoreNormalizer(self.state_dim, self.dvc)

        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,256)).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr, weight_decay=1e-5)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,256),(self.net_width,128)).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr, weight_decay=1e-5)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.seq_len, max_size=int(1e5), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            # 修复：确保target_entropy在正确的设备上且不需要梯度
            self.target_entropy = torch.tensor(-self.action_dim, dtype=torch.float32, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            # 修复：确保log_alpha在正确的设备上
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=self.dvc)
            self.alpha_lr = self.c_lr * 0.1  # 设置alpha的学习率为critic学习率的10%
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            # self.min_alpha = 0.02

    def select_action(self, prev_state, prev_action, current_state, deterministic, total_steps, env_id=None):
        # 单步模式：prev_state: (state_dim,), prev_action: (action_dim,), current_state: (state_dim,)
        with torch.no_grad():
            prev_state = torch.FloatTensor(prev_state[np.newaxis, ...]).to(self.dvc, non_blocking=True)  # (1, state_dim)
            prev_action = torch.FloatTensor(prev_action[np.newaxis, ...]).to(self.dvc, non_blocking=True)  # (1, action_dim)
            current_state = torch.FloatTensor(current_state[np.newaxis, ...]).to(self.dvc, non_blocking=True)  # (1, state_dim)
            
            # 应用Z-score标准化
            prev_state = self.state_normalizer.normalize(prev_state)
            current_state = self.state_normalizer.normalize(current_state)
            
            # 传入env_id来管理独立的隐藏状态
            env_ids = [env_id] if env_id is not None else None
            # 使用单步模式
            a, _ = self.actor(current_state=current_state, deterministic=deterministic, with_logprob=False, 
                            env_ids=env_ids, single_step_mode=True, 
                            prev_state=prev_state, prev_action=prev_action)
            
        return a.cpu().numpy()[0]

    def train(self):
        s_seq, a_seq, s_curr, a_curr, r, s_next_seq, a_next_seq, s_next_curr, dw, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # 更新训练步数
        self.training_step += 1
        
        # 更新标准化器统计量
        self.state_normalizer.update(s_seq)
        self.state_normalizer.update(s_curr)
        self.state_normalizer.update(s_next_seq)
        self.state_normalizer.update(s_next_curr)
        
        # 对状态进行标准化
        s_seq = self.state_normalizer.normalize(s_seq)
        s_curr = self.state_normalizer.normalize(s_curr)
        s_next_seq = self.state_normalizer.normalize(s_next_seq)
        s_next_curr = self.state_normalizer.normalize(s_next_curr)

        #----------------------------- Update Q Net ------------------------------#
        with torch.no_grad():
            # 修正：使用序列模式调用actor
            a_next, log_pi_a_next = self.actor(state_sequence=s_next_seq, action_sequence=a_next_seq, 
                                              current_state=s_next_curr, deterministic=False, with_logprob=True, 
                                              single_step_mode=False)
            # 修正：使用序列模式调用target critic
            target_Q1, target_Q2 = self.q_critic_target(state_sequence=s_next_seq, action_sequence=a_next_seq, 
                                                        current_state=s_next_curr, current_action=a_next,
                                                        single_step_mode=False)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next)

        # 修正：使用序列模式调用critic
        current_Q1, current_Q2 = self.q_critic(state_sequence=s_seq, action_sequence=a_seq, 
                                               current_state=s_curr, current_action=a_curr,
                                               single_step_mode=False)
        td_error = (current_Q1 - target_Q).detach().abs() + (current_Q2 - target_Q).detach().abs()

        loss_Q1 = F.mse_loss(current_Q1, target_Q, reduction='none')
        loss_Q2 = F.mse_loss(current_Q2, target_Q, reduction='none')
        q_loss = (loss_Q1 + loss_Q2) * weights
        q_loss = q_loss.mean()

        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.max_grad_norm)
        self.q_critic_optimizer.step()
        
        self.q_loss_history.append(q_loss.item())

        # 更新采样优先级
        new_priorities = td_error.mean(dim=1)
        self.replay_buffer.update_priorities(indices, new_priorities)

        # 预热阶段：只更新critic，不更新actor和alpha
        if self.training_step <= self.critic_warmup_steps:
            #----------------------------- Update Target Net (仅在预热期间) ------------------------------#
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # 预热期间返回空的actor和alpha loss
            return {
                'q_loss': q_loss.item(),
                'a_loss': 0.0,  # 预热期间不更新actor
                'alpha_loss': 0.0,  # 预热期间不更新alpha
                'alpha': self.alpha.item() if (self.adaptive_alpha and hasattr(self.alpha, 'item')) else self.alpha,
                'warmup': True,  # 标识当前处于预热阶段
                'warmup_progress': self.training_step / self.critic_warmup_steps
            }

        # 预热后，累计critic更新次数
        self.critic_update_count += 1
        
        # 检查是否需要更新actor和alpha（延迟更新）
        should_update_policy = (self.critic_update_count % self.policy_update_freq == 0)
        
        a_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        
        if should_update_policy:
            #----------------------------- Update Actor Net (延迟更新) ------------------------------#
            # 重要：确保创建新的计算图，不依赖之前的计算
            for params in self.q_critic.parameters(): 
                params.requires_grad = False

            # 修改：重新运行一次前向传播，确保新的计算图
            with torch.set_grad_enabled(True):  # 显式启用梯度计算
                # 修正：使用序列模式调用actor
                a_policy, log_pi_a = self.actor(state_sequence=s_seq, action_sequence=a_seq, 
                                               current_state=s_curr, deterministic=False, with_logprob=True,
                                               single_step_mode=False)
                # 修正：使用序列模式调用critic
                current_Q1, current_Q2 = self.q_critic(state_sequence=s_seq, action_sequence=a_seq, 
                                                       current_state=s_curr, current_action=a_policy,
                                                       single_step_mode=False)
                Q = torch.min(current_Q1, current_Q2)
                a_loss = (self.alpha * log_pi_a - Q).mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            self.a_loss_history.append(a_loss.item())

            # 解冻critic网络的参数
            for params in self.q_critic.parameters(): 
                params.requires_grad = True

            #----------------------------- Update alpha (延迟更新) ------------------------------#
            if self.adaptive_alpha:
                # 修复：确保所有张量在同一设备上
                # 关键修改：完全独立运行actor，创建新的计算图
                with torch.set_grad_enabled(True):  # 显式启用梯度计算
                    # 重要：完全独立的前向传播，不共享任何计算图
                    # 修正：使用序列模式调用actor
                    _, log_pi_a_for_alpha = self.actor(state_sequence=s_seq, action_sequence=a_seq, 
                                                      current_state=s_curr, deterministic=False, with_logprob=True,
                                                      single_step_mode=False)
                    
                    # 修复：确保target_entropy在正确的设备上并创建一个detach的副本
                    target_entropy = self.target_entropy.detach()
                    alpha_loss = -(self.log_alpha * (log_pi_a_for_alpha + target_entropy).detach()).mean()

                # 修复：在优化之前清零梯度并确保log_alpha在正确设备上
                self.alpha_optim.zero_grad()
                
                # 检查设备一致性
                if self.log_alpha.device != self.dvc:
                    self.log_alpha = self.log_alpha.to(self.dvc)
                
                alpha_loss.backward()  
                torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
                self.alpha_optim.step()
                
                # 更新alpha，移除最小值约束，允许完全自动调节
                self.alpha = self.log_alpha.exp()
                
                # 移除alpha最小值限制的同步代码
                # 现在alpha可以自由调节到任何正值
                
                self.alpha_loss_history.append(alpha_loss.item())
            else:
                alpha_loss = torch.tensor(0.0)
                self.alpha_loss_history.append(0.0)
        else:
            # 不更新actor和alpha时，添加占位符loss
            self.a_loss_history.append(0.0)
            self.alpha_loss_history.append(0.0)

        #----------------------------- Update Target Net ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'a_loss': a_loss.item(),
            'alpha_loss': alpha_loss.item() if self.adaptive_alpha else 0.0,
            # 修复：统一处理alpha值的获取
            'alpha': self.alpha.item() if (self.adaptive_alpha and hasattr(self.alpha, 'item')) else self.alpha,
            'warmup': False,  # 标识预热阶段已完成
            'policy_updated': should_update_policy,  # 标识是否更新了策略
            'critic_update_count': self.critic_update_count,  # 返回critic更新次数
            'should_update_policy': should_update_policy  # 新增：明确标识是否应该更新策略
        }

    def save(self,EnvName, timestep):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))
        # 保存Z-score标准化器的参数
        torch.save(self.state_normalizer.state_dict(), "./model/{}_normalizer{}.pth".format(EnvName,timestep))

    def load(self,EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))
        # 加载Z-score标准化器的参数
        try:
            normalizer_state = torch.load("./model/{}_normalizer{}.pth".format(EnvName, timestep), map_location=self.dvc)
            self.state_normalizer.load_state_dict(normalizer_state)
        except FileNotFoundError:
            print("Warning: Normalizer parameters not found, using default initialization.")


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, seq_len, max_size, dvc, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0
        self.seq_len = seq_len
        
        # 存储序列数据
        self.s_seq = torch.zeros((max_size, seq_len, state_dim), dtype=torch.float, device=self.dvc)  # 历史状态序列
        self.a_seq = torch.zeros((max_size, seq_len, action_dim), dtype=torch.float, device=self.dvc)  # 历史动作序列
        self.s_curr = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)  # 当前状态
        self.a_curr = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)  # 当前动作
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)  # 奖励
        self.s_next_seq = torch.zeros((max_size, seq_len, state_dim), dtype=torch.float, device=self.dvc)  # 下一个状态的历史序列
        self.a_next_seq = torch.zeros((max_size, seq_len, action_dim), dtype=torch.float, device=self.dvc)  # 下一个状态对应的动作序列
        self.s_next_curr = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)  # 下一个当前状态
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)
        
        # 初始化所有样本的优先级（初始值设为1）
        self.priorities = torch.ones((max_size, 1), dtype=torch.float, device=self.dvc)
        # PER参数
        self.alpha = alpha   # 控制优先级的重要性
        self.beta = beta     # importance-sampling权重
        self.epsilon = 1e-6  # 防止优先级为0

    def add(self, s_seq, a_seq, s_curr, action, r, s_next_seq, a_next_seq, s_next_curr, dw):
        # s_seq: (seq_len, state_dim), a_seq: (seq_len, action_dim)
        # s_curr: (state_dim,), action: (action_dim,), s_next_curr: (state_dim,)
        # a_next_seq: (seq_len, action_dim) - 下一个状态对应的动作序列
        self.s_seq[self.ptr] = torch.from_numpy(s_seq).to(self.dvc, non_blocking=True)
        self.a_seq[self.ptr] = torch.from_numpy(a_seq).to(self.dvc, non_blocking=True)
        self.s_curr[self.ptr] = torch.from_numpy(s_curr).to(self.dvc, non_blocking=True)
        self.a_curr[self.ptr] = torch.from_numpy(action).to(self.dvc, non_blocking=True)
        self.r[self.ptr] = r
        self.s_next_seq[self.ptr] = torch.from_numpy(s_next_seq).to(self.dvc, non_blocking=True)
        self.a_next_seq[self.ptr] = torch.from_numpy(a_next_seq).to(self.dvc, non_blocking=True)  # 存储下一个动作序列
        self.s_next_curr[self.ptr] = torch.from_numpy(s_next_curr).to(self.dvc, non_blocking=True)
        self.dw[self.ptr] = bool(dw)
        
        # 对新加入的样本赋予最大的优先级
        if self.size > 0:
            max_prio = self.priorities[:self.size].max().item()
        else:
            max_prio = 1.0
        self.priorities[self.ptr] = max_prio

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 从当前样本中根据优先级采样
        prios = self.priorities[:self.size]  # (size,1)
        
        # 修复：处理大数组的内存问题
        # 将计算移到CPU上，避免GPU内存不足
        prios_cpu = prios.cpu().flatten()  # 转换到CPU并展平
        probs_cpu = (prios_cpu + self.epsilon) ** self.alpha
        probs_cpu = probs_cpu / probs_cpu.sum()
        
        # 修复：将tensor转换为numpy数组再检查NaN和inf
        probs_numpy = probs_cpu.numpy()
        if np.any(np.isnan(probs_numpy)) or np.any(np.isinf(probs_numpy)):
            probs_numpy = np.ones(self.size) / self.size
            
        # 修复：确保batch_size不超过当前样本数
        actual_batch_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, actual_batch_size, p=probs_numpy)
        indices = torch.tensor(indices, device=self.dvc)

        # importance-sampling权重 - 在CPU上计算，然后转回GPU
        total = self.size
        selected_probs = torch.tensor(probs_numpy[indices.cpu()], device=self.dvc).view(-1, 1)
        weights = (total * selected_probs) ** (-self.beta)
        weights = weights / weights.max()

        return (self.s_seq[indices], self.a_seq[indices], self.s_curr[indices], self.a_curr[indices], 
                self.r[indices], self.s_next_seq[indices], self.a_next_seq[indices], self.s_next_curr[indices], 
                self.dw[indices], indices, weights)

    def update_priorities(self, indices, new_priorities):
        # 更新采样的优先级值，new_priorities可以是TD error的绝对值
        new_priorities = new_priorities.view(-1, 1)
        self.priorities[indices] = new_priorities + self.epsilon
        
    def update_beta(self, progress):
        # 更新beta值
        self.beta = 0.4 + progress * (1.0 - 0.4)  # 逐渐增加beta值