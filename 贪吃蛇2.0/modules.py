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

# class DuelingDQN(nn.Module):
#     def __init__(self, n_act):
#         super(DuelingDQN, self).__init__()
#         # Convolutional layers
#         # self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=5, stride=5)
#         # self.conv2 = torch.nn.Conv2d(18, 36, kernel_size=3, stride=2)
#         # self.conv3 = torch.nn.Conv2d(36, 72, kernel_size=2, stride=1)
#         self.relu = nn.LeakyReLU()
#         # Flatten followed by common dense layer
#         self.fc = torch.nn.Linear(10*10, 512)
#         self.dropout = nn.Dropout(0.2)
#         # Value stream
#         self.value_fc1 = nn.Linear(512, 512)
#         self.value_fc2 = nn.Linear(512, 512)
#         self.value_fc3 = nn.Linear(512, 512)
#         self.value_fc4 = nn.Linear(512, 256)
#         self.value_fc5 = nn.Linear(256, 128)  # This outputs a single value: V(s)
#         self.value = nn.Linear(128, 1)  # This outputs a single value: V(s)
#
#         # Advantage stream
#         self.advantage_fc1 = nn.Linear(512, 512)
#         self.advantage_fc2 = nn.Linear(512, 512)
#         self.advantage_fc3 = nn.Linear(512, 512)
#         self.advantage_fc4 = nn.Linear(512, 256)
#         self.advantage_fc5 = nn.Linear(256, 128)
#         self.advantage = nn.Linear(128, n_act)  # This outputs advantages for each action: A(s,a)
#
#     def forward(self, x):
#         # # Convolutional layers
#         # x = F.leaky_relu_(self.conv1(x))
#         # x = F.leaky_relu_(self.conv2(x))
#         # x = F.leaky_relu_(self.conv3(x))
#         #
#         # # Flatten
#         #x = x.view(x.size(0), -1)
#         # Common dense layer
#         x = self.fc(x)
#         x = self.relu(x)
#         # Dueling Part: Value and Advantage streams
#         value = self.value_fc1(x)
#         value = self.relu(value)
#         value = self.value_fc2(value)
#         value = self.relu(value)
#         value = self.value_fc3(value)
#         value = self.relu(value)
#         value = self.value_fc4(value)
#         value = self.relu(value)
#         value = self.value_fc5(value)
#         value = self.relu(value)
#         value = self.value(value)  # V(s)
#
#         advantage = self.advantage_fc1(x)
#         advantage = self.relu(advantage)
#         advantage = self.advantage_fc2(advantage)
#         advantage = self.relu(advantage)
#         advantage = self.advantage_fc3(advantage)
#         advantage = self.relu(advantage)
#         advantage = self.advantage_fc4(advantage)
#         advantage = self.relu(advantage)
#         advantage = self.advantage_fc5(advantage)
#         advantage = self.relu(advantage)
#         advantage = self.advantage(advantage)  # A(s,a)
#
#         # Combine V(s) and A(s,a) to get Q values
#         qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return qvals
# class DuelingDQN(nn.Module):
#     def __init__(self, n_act,mid_dim=600):
#         super(DuelingDQN, self).__init__()
#         self.relu = nn.LeakyReLU()
#         # Flatten followed by common dense layer
#         self.fc = torch.nn.Linear(20, mid_dim)
#         self.dropout = nn.Dropout(0.2)
#         # Value stream
#         self.value_fc1 = nn.Linear(mid_dim, mid_dim)
#         self.value_fc2 = nn.Linear(mid_dim, mid_dim)
#         self.value_fc3 = nn.Linear(mid_dim, mid_dim)
#         self.value_fc4 = nn.Linear(mid_dim, mid_dim)
#         self.value_fc5 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
#         self.value_fc6 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
#         self.value_fc7 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
#         self.value = nn.Linear(mid_dim, 1)  # This outputs a single value: V(s)
#
#
        # # Advantage stream
        # self.advantage_fc1 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc2 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc3 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc4 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc5 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc6 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc7 = nn.Linear(mid_dim, mid_dim)
        # self.advantage = nn.Linear(mid_dim, n_act)  # This outputs advantages for each action: A(s,a)
#
#     def forward(self, x):
#         # Common dense layer
#         x = self.fc(x)
#         x = self.relu(x)
#         # Dueling Part: Value and Advantage streams
#         value = self.value_fc1(x)
#         value = self.relu(value)
#         value = self.value_fc2(value)
#         value = self.relu(value)
#         value = self.value_fc3(value)
#         value = self.relu(value)
#         value = self.dropout(value)
#         value = self.value_fc4(value)
#         value = self.relu(value)
#         value = self.value_fc5(value)
#         value = self.relu(value)
#         value = self.value_fc6(value)
#         value = self.relu(value)
#         value = self.dropout(value)
#         value = self.value_fc7(value)
#         value = self.relu(value)
#         value = self.value(value)  # V(s)
#
        # advantage = self.advantage_fc1(x)
        # advantage = self.relu(advantage)
        # advantage = self.advantage_fc2(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.advantage_fc3(advantage)
        # advantage = self.dropout(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.advantage_fc4(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.advantage_fc5(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.advantage_fc6(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.dropout(advantage)
        # advantage = self.advantage_fc7(advantage)
        # advantage = self.relu(advantage)
        # advantage = self.advantage(advantage)  # A(s,a)
#
#         # Combine V(s) and A(s,a) to get Q values
#         qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return qvals

class Actor(nn.Module):
    def __init__(self, obs_size, n_act, mid_dim=512):
        super(Actor, self).__init__()
        self.relu = nn.LeakyReLU()
        # Flatten followed by common dense layer
        self.fc = torch.nn.Linear(obs_size, mid_dim)
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
        value = self.value(value)
        return value

class Critic(nn.Module):
    def __init__(self, obs_size, n_act, mid_dim=600):
        super(Critic, self).__init__()
        self.relu = nn.LeakyReLU()
        # Flatten followed by common dense layer
        self.fc = torch.nn.Linear(obs_size, mid_dim)
        self.dropout = nn.Dropout(0.2)
        # Advantage stream
        self.advantage_fc1 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc2 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc3 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc4 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc5 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc6 = nn.Linear(mid_dim, mid_dim)
        # self.advantage_fc7 = nn.Linear(mid_dim, mid_dim)
        self.advantage = nn.Linear(mid_dim, n_act)  # This outputs advantages for each action: A(s,a)

    def forward(self,x):
        # Common dense layer
        x = self.fc(x)
        x = self.relu(x)
        advantage = self.advantage_fc1(x)
        advantage = self.relu(advantage)
        advantage = self.advantage_fc2(advantage)
        advantage = self.relu(advantage)
        advantage = self.advantage_fc3(advantage)
        advantage = self.dropout(advantage)
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
        return advantage


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

actor = Actor(n_act=4,obs_size=20)  # 示例：n_act设置为4
critic = Critic(n_act=4,obs_size=20)  # 示例：n_act设置为4
print(f"The model has {count_parameters(actor):,} trainable parameters.")
print(f"The model has {count_parameters(critic):,} trainable parameters.")