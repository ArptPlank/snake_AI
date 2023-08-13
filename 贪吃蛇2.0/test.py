import torch
import gameEnvironment
from tqdm import tqdm
import modules
import matplotlib.pyplot as plt
import agentTest
class Test():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gameEnvironment.game()
        self.env.fps = 10
        n_act = self.env.n_act
        n_obs = self.env.n_obs
        actor = modules.Actor(n_obs,n_act)
        critic = modules.Critic(n_obs,n_act)
        actor.load_state_dict(torch.load('./model/actor/trained_model_39000.pth'))
        critic.load_state_dict(torch.load('./model/critic/trained_model_39000.pth'))
        self.agent = agentTest.DQNAgent(
            actor=actor,
            critic=critic,
            n_act=n_act,
        )

    def test_episode(self,episodes):
        total_reward = 0
        num = 0
        obs = self.env.reset()
        # obs = torch.tensor(obs).to(self.device)
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, length = self.env.step(action, episodes)
            total_reward += reward
            obs = torch.tensor(next_obs).to(self.device)
            obs = next_obs
            self.env.render(True, episodes+1)
            num += 1
            if done: break
        return total_reward, length, num

    def test(self):
        lengths = []
        live = []
        for epoch in tqdm(range(150)):
            total_reward,length,num = self.test_episode(episodes=epoch)
            lengths.append(length)
            live.append(num)
        #length用于画图,记录每一轮的长度
        return lengths,live

    #用length列表画饼状图，表示各种长度的占比
    def bucket_data(self,data, interval=10):
        """Bucket data into intervals."""
        min_val = min(data)
        max_val = max(data)
        buckets = [(i, i + interval) for i in range(min_val, max_val + 1, interval)]

        bucketed_data = {}
        for start, end in buckets:
            count = sum(1 for x in data if start <= x < end)
            label = f"{start}-{end - 1}"
            bucketed_data[label] = count

        return list(bucketed_data.keys()), list(bucketed_data.values())

    def plot_pie_charts(self, lengths, live):
        # For lengths data
        unique_lengths = set(lengths)
        frequencies_lengths = [lengths.count(val) for val in unique_lengths]
        avg_length_value = sum(lengths) / len(lengths)

        # For live data
        live_labels, live_frequencies = self.bucket_data(live)
        avg_live_value = sum(live) / len(live)

        # Create subplots for the two pie charts
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plotting for lengths
        axes[0].pie(frequencies_lengths, labels=unique_lengths, autopct='%1.1f%%', startangle=140)
        axes[0].set_title('Proportion of Each Value in "lengths" List')
        axes[0].text(0, -1.2, 'Average Value: {:.2f}'.format(avg_length_value), ha='center')

        # Plotting for live
        axes[1].pie(live_frequencies, labels=live_labels, autopct='%1.1f%%', startangle=140)
        axes[1].set_title('Distribution of "live" List in Intervals of 5')
        axes[1].text(0, -1.2, 'Average Value: {:.2f}'.format(avg_live_value), ha='center')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    test = Test()
    lengths,live = test.test()
    test.plot_pie_charts(lengths, live)