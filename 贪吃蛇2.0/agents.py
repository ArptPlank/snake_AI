import copy
import numpy as np
import torch
import torchUtils
import preprocess
import numpy
class DQNAgent(object):

    def __init__(self, actor,critic, actor_optimizer,critic_optimizer,replay_buffer,batch_size,replay_start_size, n_act,update_target_steps,explorer,actor_scheduler,critic_scheduler,gamma=0.9):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Running on", self.device)
        self.explorer = explorer
        self.actor_pred_func = actor #价值函数
        self.actor_target_func = copy.deepcopy(actor) #价值函数
        self.actor_pred_func.to(self.device)
        self.actor_target_func.to(self.device)
        self.actor_optimizer = actor_optimizer  # 价值函数优化器
        self.actor_scheduler = actor_scheduler  # 价值函数学习率衰减率
        self.critic_pred_func = critic #策略含数
        self.critic_target_func = copy.deepcopy(critic) #策略函数
        self.critic_pred_func.to(self.device)
        self.critic_target_func.to(self.device)
        self.critic_optimizer = critic_optimizer  # 策略函数优化器
        self.critic_scheduler = critic_scheduler  # 策略函数学习率衰减率
        self.criterion = torch.nn.SmoothL1Loss()  # 损失函数
        self.rb = replay_buffer
        self.update_target_steps = update_target_steps
        self.global_step = 0
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.n_act = n_act  # 动作数量
        self.gamma = gamma  # 收益衰减率

    # 根据经验得到action
    def predict(self, obs):
        #obs = torch.FloatTensor(obs)
        obs = obs.float()
        Q_values = self.take_action_pred(obs)
        # action = int(torch.argmax(Q_values).detach().numpy())
        action = torch.argmax(Q_values).item()
        return action

    # 根据探索与利用得到action
    def act(self, obs,episodes):
        return self.explorer.act(self.predict, obs, episodes)

    # 更新Q表格
    def learn(self, obs, action, reward, next_obs, done, num):
        self.global_step += 1
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        self.rb.append((obs.squeeze(), action, reward, next_obs.squeeze(), done))
        if len(self.rb) > self.replay_start_size and self.global_step%self.rb.num_steps >= 0 and num == 0:
            self.learn_batch(*self.rb.sample(self.batch_size))
        if self.global_step % self.update_target_steps == 0:
            self.syns_target()

    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):
        batch_obs = batch_obs.to(self.device)
        batch_action = batch_action.to(self.device)
        batch_reward = batch_reward.to(self.device)
        batch_next_obs = batch_next_obs.to(self.device)
        batch_done = batch_done.to(self.device)
        pred_Vs = self.take_action_pred(batch_obs)
        action_onehot = torchUtils.one_hot(batch_action,self.n_act)
        predict_Q = (pred_Vs*action_onehot).sum(dim=1)
        target_Q = batch_reward + (1 - batch_done.float()) * self.gamma * self.take_action_target(batch_next_obs).max(1)[0]
        # 更新参数
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.actor_scheduler.step(loss)
        self.critic_scheduler.step(loss)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def syns_target(self):
        for target_parma,parma in zip(self.actor_target_func.parameters(),self.actor_pred_func.parameters()):
            target_parma.data.copy_(parma.data)
        for target_parma,parma in zip(self.critic_target_func.parameters(),self.critic_pred_func.parameters()):
            target_parma.data.copy_(parma.data)

    def take_action_pred(self, obs):
        value = self.actor_pred_func(obs)
        advantage = self.critic_pred_func(obs)
        #Combine V(s) and A(s,a) to get Q values
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals

    def take_action_target(self, obs):
        value = self.actor_target_func(obs)
        advantage = self.critic_target_func(obs)
        #Combine V(s) and A(s,a) to get Q values
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals