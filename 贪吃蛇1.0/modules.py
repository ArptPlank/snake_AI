import torch
from torch import nn
import torch.nn.functional as F
class MLP(torch.nn.Module):

    def __init__(self, obs_size,n_act):
        super().__init__()
        self.mlp = self.__mlp(obs_size,n_act)

    def __mlp(self,obs_size,n_act):
        return torch.nn.Sequential(
            torch.nn.Linear(obs_size, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, n_act)
        )

    def forward(self, x):
        return self.mlp(x)

class DuelingDQN(nn.Module):
    def __init__(self, n_act,mid_dim=512):
        super(DuelingDQN, self).__init__()
        self.relu = nn.LeakyReLU()
        # Flatten followed by common dense layer
        self.fc = torch.nn.Linear(18, mid_dim)
        self.dropout = nn.Dropout(0.2)
        # Value stream
        self.value_fc1 = nn.Linear(mid_dim, mid_dim)
        self.value_fc2 = nn.Linear(mid_dim, mid_dim)
        self.value_fc3 = nn.Linear(mid_dim, mid_dim)
        self.value_fc4 = nn.Linear(mid_dim, mid_dim)
        self.value_fc5 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
        # self.value_fc6 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
        # self.value_fc7 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
        self.value = nn.Linear(mid_dim, 1)  # This outputs a single value: V(s)


        # Advantage stream
        self.advantage_fc1 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc2 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc3 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc4 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc5 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc6 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc7 = nn.Linear(mid_dim, mid_dim)
        self.advantage = nn.Linear(mid_dim, n_act)  # This outputs advantages for each action: A(s,a)

    def forward(self, x):
        # Common dense layer
        x = self.fc(x)
        x = self.relu(x)
        # Dueling Part: Value and Advantage streams
        value = self.value_fc1(x)
        value = self.relu(value)
        value = self.value_fc2(value)
        value = self.relu(value)
        value = self.value_fc3(value)
        value = self.relu(value)
        value = self.dropout(value)
        value = self.value_fc4(value)
        value = self.relu(value)
        value = self.value_fc5(value)
        value = self.relu(value)
        # value = self.value_fc6(value)
        # value = self.relu(value)
        # value = self.dropout(value)
        # value = self.value_fc7(value)
        # value = self.relu(value)
        value = self.value(value)  # V(s)

        advantage = self.advantage_fc1(x)
        advantage = self.relu(advantage)
        advantage = self.advantage_fc2(advantage)
        advantage = self.relu(advantage)
        advantage = self.advantage_fc3(advantage)
        value = self.dropout(value)
        advantage = self.relu(advantage)
        advantage = self.advantage_fc4(advantage)
        advantage = self.relu(advantage)
        advantage = self.advantage_fc5(advantage)
        advantage = self.relu(advantage)
        # advantage = self.advantage_fc6(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.dropout(advantage)
        # advantage = self.advantage_fc7(advantage)
        # advantage = self.relu(advantage)
        advantage = self.advantage(advantage)  # A(s,a)

        # Combine V(s) and A(s,a) to get Q values
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

dueling_dqn = DuelingDQN(n_act=4)  # 示例：n_act设置为4
print(f"The model has {count_parameters(dueling_dqn):,} trainable parameters.")