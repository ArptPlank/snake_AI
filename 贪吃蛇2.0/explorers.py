import numpy as np

class EpsilonGreedy():
    def __init__(self,n_act,e_greed,decay_rate,num_decay):
        self.n_act = n_act
        self.epsilon = e_greed
        self.decay = decay_rate
        self.num_decay = num_decay

    def act(self,predict_method,obs,episodes):
        if np.random.uniform(0, 1) < self.epsilon:  #探索
            action = np.random.choice(self.n_act)
        else: # 利用
            action = predict_method(obs)
        if episodes % self.num_decay == 0:
            self.epsilon = max(0.03, self.epsilon - self.decay)
        return action