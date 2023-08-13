import copy
import torch
class DQNAgent(object):

    def __init__(self, q_func,n_act):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Running on", self.device)
        self.pred_func = q_func #Q函数
        #self.target_func = copy.deepcopy(q_func)
        self.pred_func.to(self.device)
        #self.target_func.to(self.device)
        #self.global_step = 0
        self.n_act = n_act  # 动作数量

    # 根据经验得到action
    def predict(self, obs):
        obs = obs.float()
        Q_values = self.pred_func(obs)
        action = torch.argmax(Q_values).item()
        return action