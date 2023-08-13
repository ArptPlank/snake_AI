import copy
import numpy as np
import torch
import torchUtils
import preprocess
import numpy
class DQNAgent(object):

    def __init__(self,n_act,actor,critic):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Running on", self.device)
        self.actor_pred_func = actor  # 价值函数
        self.actor_pred_func.to(self.device)
        self.critic_pred_func = critic  # 策略含数
        self.critic_pred_func.to(self.device)
        self.global_step = 0
        self.n_act = n_act  # 动作数量

    # 根据经验得到action
    def predict(self, obs):
        #obs = torch.FloatTensor(obs)
        obs = obs.float()
        Q_values = self.take_action_pred(obs)
        # action = int(torch.argmax(Q_values).detach().numpy())
        action = torch.argmax(Q_values).item()
        return action

    def take_action_pred(self, obs):
        value = self.actor_pred_func(obs)
        advantage = self.critic_pred_func(obs)
        #Combine V(s) and A(s,a) to get Q values
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals