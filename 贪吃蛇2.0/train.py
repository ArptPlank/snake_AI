import agents,modules,replay_buffers,explorers
import torch
import time
import gameEnvironment
import preprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
class TrainManager():

    def __init__(self,
                 env,  #环境
                 episodes=1000,  #轮次数量
                 batch_size=64,  #每一批次的数量
                 num_steps=4,  #进行学习的频次
                 memory_size = 2000,  #经验回放池的容量
                 replay_start_size = 200,  #开始回放的次数
                 actor_lr=0.001,  #价值函数学习率
                 critic_lr=0.001,  #策略函数学习率
                 update_target_steps=200,
                 gamma=0.9,  #收益衰减率
                 e_greed=0.3 , #探索与利用中的探索概率
                 decay_rate=0.001,#探索概率衰减
                 num_decay=10,#探索概率衰减频率
                 factor=0.8,#学习率减小的比例
                 patience=500,#在指标停止改善后，等待多少个 epoch 后降低学习率
                 threshold=1e-6,#判断指标是否停止改善的阈值
                 min_lr=1e-6#学习率可以降低到的最小值
                 ):
        self.env = env
        self.episodes = episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # n_act = env.action_space.n
        # n_obs = env.observation_space.shape[0]
        n_act = env.n_act
        n_obs = env.n_obs
        #q_func = modules.MLP(n_obs, n_act)
        actor = modules.Actor(obs_size=n_obs,n_act=n_act)
        critic = modules.Critic(obs_size=n_obs,n_act=n_act)
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=critic_lr)
        rb = replay_buffers.ReplayBuffer(memory_size,num_steps)
        explorer = explorers.EpsilonGreedy(n_act,e_greed,decay_rate,num_decay)
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='min', factor=factor, patience=patience,threshold=threshold, min_lr=min_lr)
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', factor=factor, patience=patience,threshold=threshold, min_lr=min_lr)
        self.agent = agents.DQNAgent(
            actor=actor,
            actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler,
            critic=critic,
            critic_optimizer=critic_optimizer,
            critic_scheduler=critic_scheduler,
            replay_buffer = rb,
            batch_size=batch_size,
            replay_start_size = replay_start_size,
            n_act=n_act,
            gamma=gamma,
            update_target_steps=update_target_steps,
            explorer=explorer,
        )

    # 训练一轮游戏
    def train_episode(self,episodes):
        total_reward = 0
        obs = self.env.reset()
        # obs = torch.tensor(obs).to(self.device)
        num = 0
        isdraw = False
        # if episodes >= 10000:
        #     isdraw = True
        # else:
        #     isdraw = False
        while True:
            action = self.agent.act(obs,episodes)
            next_obs, reward, done,length = self.env.step(action,episodes)
            total_reward += reward
            next_obs = torch.tensor(next_obs).to(self.device)
            reward = torch.tensor(reward).to(self.device)
            done = torch.tensor(done).to(self.device)
            self.agent.learn(obs, action, reward, next_obs, done,num)
            obs = next_obs
            #self.env.render(isdraw,episodes)
            num += 1
            if done: break
        return total_reward,length,num

    # 测试一轮游戏
    def test_episode(self,episodes):
        total_reward = 0
        num = 0
        obs = self.env.reset()
        # obs = torch.tensor(obs).to(self.device)
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, length = self.env.step(action,episodes)
            total_reward += reward
            obs = torch.tensor(next_obs).to(self.device)
            obs = next_obs
            self.env.render(True,episodes)
            num += 1
            if done: break
        return total_reward,length,num

    def train(self):
        # 1. 开启matplotlib的交互模式
        plt.ion()
        # 2. 初始化图形
        fig, ax = plt.subplots()
        rewards = []
        line, = ax.plot(rewards)
        ax.set_title('Training Reward Over Time')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        #for e in range(self.episodes):
        split_num = int(self.episodes / 1000)
        e = 0
        t = time.time()
        sum_avr_length = 0
        for i in range(split_num):
            with tqdm(total=int(self.episodes / split_num), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.episodes / split_num)):
                    ep_reward, length, num = self.train_episode(episodes=e)
                    sum_avr_length += length
                    if e % 50 == 0:
                        rewards.append(ep_reward)
                        if len(rewards) > 1000:
                            rewards.pop(0)
                        line.set_ydata(rewards)
                        line.set_xdata(list(range(len(rewards))))
                        ax.relim()
                        ax.autoscale_view(True, True, True)
                        plt.draw()
                        plt.pause(0.001)  # 小暂停，使得图形有时间更新
                    if e % 1000 == 0:
                        actor_model_state_dict = self.agent.actor_pred_func.state_dict()
                        # 选择保存的文件路径和文件名
                        file_path = "./model/actor/trained_model_" + str(e) + (".pth")
                        # 使用torch.save()方法将参数字典保存到文件中
                        torch.save(actor_model_state_dict, file_path)
                        critic_model_state_dict = self.agent.critic_pred_func.state_dict()
                        # 选择保存的文件路径和文件名
                        file_path = "./model/critic/trained_model_" + str(e) + (".pth")
                        # 使用torch.save()方法将参数字典保存到文件中
                        torch.save(critic_model_state_dict, file_path)
                    if e % 50 == 0:
                        avr_length = sum_avr_length / 50
                        sum_avr_length = 0
                        actor_lr = self.agent.actor_optimizer.param_groups[0]['lr']
                        critic_lr = self.agent.critic_optimizer.param_groups[0]['lr']
                        pbar.set_postfix({'episode': '%d' % e, 'reward': '%.3f' % ep_reward,'length': '%.6f' % avr_length, 'step_num': '%.3f' % num,'actor_lr': '%.8f' % actor_lr,'critic_lr':'%.8f' %critic_lr,'e_greed':'%.3f' %self.agent.explorer.epsilon,'time': '%.3f' % (time.time() - t)})
                        t = time.time()
                    # if e % 1000 == 0:
                    #     for epoch in range(10):
                    #         self.test_episode(episodes=e)
                    e += 1
                    pbar.update(1)


if __name__ == '__main__':
    #env1 = gym.make("MountainCar-v0")
    #env1 = gym.make("CartPole-v0")
    env1 = gameEnvironment.game()
    tm = TrainManager(env1,update_target_steps=200,episodes=100000,actor_lr=6e-3,critic_lr=3e-2,batch_size=512,memory_size=5000,replay_start_size=1024,decay_rate=0.00002,e_greed=0.40,num_decay=20,gamma=0.9,num_steps=20,min_lr=1e-8,factor=0.95,patience=200,threshold=1e-7)
    tm.train()