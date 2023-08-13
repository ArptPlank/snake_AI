# 导入pygame模块
import pygame
# 导入random模块
import random
import math
from torch import nn
from torch import zeros
from torch import device
from torch.cuda import is_available
from torch import argmax
from torch import load
import tkinter as tk
from tkinter import simpledialog
import sys
import os
# 定义一个蛇类
class Snake:
    # 初始化方法，传入蛇的位置，颜色和大小参数
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    # 定义一个绘制方法，传入窗口对象参数
    def draw(self, window,color):
        # 使用pygame.draw.rect方法绘制一个矩形代表蛇的身体
        pygame.draw.rect(window, color, [self.x, self.y, self.size, self.size])
# 定义一个食物类
class Food:
    # 初始化方法，传入食物的位置，颜色和大小参数
    def __init__(self, x, y, color, size):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
    # 定义一个绘制方法，传入窗口对象参数
    def draw(self, window):
        # 使用pygame.draw.rect方法绘制一个矩形代表食物
        pygame.draw.rect(window, self.color, [self.x, self.y, self.size, self.size])

class game():
    def __init__(self,fps,device,sound_path):
        self.epoch = 1
        self.device = device
        # 初始化pygame
        pygame.init()
        # 加载音频
        self.sound = pygame.mixer.Sound(sound_path)
        # 初始化游戏窗口尺寸
        self.size = 10
        # 设置窗口的大小和标题
        self.window_width = 40 * self.size
        self.window_height = 40 * self.size
        self.window = pygame.display.set_mode((self.window_width, self.window_height),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.window.fill((255, 255, 255))  # 设置背景颜色为白色
        pygame.display.set_caption("Snake Game")
        # 设置颜色常量
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0,0,255)
        # 设置蛇和食物的大小
        self.snake_size = 40
        self.food_size = 40
        # 设置蛇的初始位置和方向
        self.snake_x = self.size // 2 * 40
        self.snake_y = self.size // 2 * 40
        self.snake_dx = 0
        self.snake_dy = 0
        # 设置蛇的初始长度和身体列表
        self.snake_length = 2
        self.snake_body = []
        snake = Snake(self.snake_x, self.snake_y,self.snake_size)
        self.snake_body.append(snake)
        snake2 = Snake(self.snake_x-self.snake_size,self.snake_y,self.snake_size)
        self.snake_body.append(snake2)
        # 设置食物的初始位置
        self.food_x = random.randint(0, self.size-1) * 40
        self.food_y = random.randint(0, self.size-1) * 40
        # 设置游戏的初始状态和分数
        self.game_over = False
        self.score = 0
        self.clock = pygame.time.Clock()
        # 设置字体对象和大小
        self.font = pygame.font.SysFont("arial", 32)
        # 创建一个食物对象，传入食物的位置，颜色和大小参数
        self.food_ = Food(self.food_x, self.food_y, self.red, self.food_size)
        self.n_act = 4
        self.n_obs = 18
        self.last_action = 4
        self.distance = math.sqrt(((self.snake_body[0].x-self.food_x)/40)**2+((self.snake_body[0].y-self.food_y)/40)**2)
        self.base_distance = 3
        self.len_num = 0
        self.step_num = 0
        # 设置游戏的帧率和时钟对象
        self.fps = fps
        # 计算过渡颜色
        self.segment_color = self.linear_gradient(self.blue, self.white, 30)
    def step(self,action):
        self.len_num += 1
        self.step_num += 1
        # action 0--上   1--下   2--左  3--右
        if action == 0:
            if self.last_action == 1:
                self.game_over = True
            self.snake_dx = 0
            self.snake_dy = -self.snake_size
            self.len_num = 0
        elif action == 1:
            if self.last_action == 0:
                self.game_over = True
            self.snake_dx = 0
            self.snake_dy = self.snake_size
            self.len_num = 0
        elif action == 2:
            if self.last_action == 3:
                self.game_over = True
            self.snake_dx = -self.snake_size
            self.snake_dy = 0
            self.len_num = 0
        elif action == 3:
            if self.last_action == 2:
                self.game_over = True
            self.snake_dx = self.snake_size
            self.snake_dy = 0
            self.len_num = 0
        self.last_action = action
        done = False
        # 判断蛇是否超出窗口边界，如果是，设置游戏结束为真
        if self.snake_x+self.snake_dx < 0 or self.snake_x+self.snake_dx > self.window_width - self.snake_size or self.snake_y+self.snake_dy < 0 or self.snake_y+self.snake_dy > self.window_height - self.snake_size:
            self.game_over = True
        else:
            # 更新蛇的位置
            self.snake_x += self.snake_dx
            self.snake_y += self.snake_dy

        # 创建一个蛇对象，传入蛇的位置，颜色和大小参数
        snake = Snake(self.snake_x, self.snake_y,self.snake_size)
        # 将蛇对象添加到蛇身体列表的开头
        self.snake_body.insert(0, snake)
        # 如果蛇身体列表的长度大于蛇的长度，删除列表的最后一个元素
        if len(self.snake_body) > self.snake_length:
            del self.snake_body[-1]
        # 遍历蛇身体列表中除了第一个元素以外的其他元素
        for segment in self.snake_body[1:]:
            # 判断蛇是否碰到自己的身体，如果是，设置游戏结束为真
            if segment.x == snake.x and segment.y == snake.y:
                self.game_over = True

        # 判断蛇头是否与食物的位置重合
        self.iseat(snake)
        if self.game_over:
            done = True

        return self.obs_(), done,self.snake_length

    def linear_gradient(self,start_color, end_color,length):
        # 计算每种颜色的增量
        r_step = (end_color[0] - start_color[0]) / 30
        g_step = (end_color[1] - start_color[1]) / 30
        b_step = (end_color[2] - start_color[2]) / 30
        # 保存每种颜色的增量
        gradient = []
        # 计算每种颜色的增量，并保存到列表中
        for i in range(length):
            r = int(start_color[0] + r_step * i)
            g = int(start_color[1] + g_step * i)
            b = int(start_color[2] + b_step * i)
            gradient.append((r, g, b))
        return gradient
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        # 填充窗口背景颜色为黑色
        self.window.fill(self.white)
        for i, segment in enumerate(self.snake_body):
            # 如果是蛇头（即列表的第一个元素），绘制一个黄色方块
            if i == 0:
                pygame.draw.rect(self.window, self.green,
                                 (segment.x, segment.y, self.snake_size, self.snake_size))
            else:
                # 如果不是蛇头，仍然绘制一个矩形
                segment.draw(self.window, self.segment_color[i - 1])
        # 调用食物对象的绘制方法，传入窗口对象参数
        self.food_.draw(self.window)
        # 使用字体对象渲染分数文本，颜色为白色
        score_text = self.font.render("Score: " + str(self.score), True, self.black)
        # 在窗口左上角绘制分数文本
        self.window.blit(score_text, (0, 0))
        # 使用字体对象渲染帧率文本，颜色为白色
        episodes_text = self.font.render("episodes: " + str(self.epoch), True, self.black)
        # 获取帧率文本的大小以便正确地放在右上角
        episodes_text_width, episodes_text_height = self.font.size("episodes: " + str(self.epoch))
        # 在窗口右上角绘制帧率文本
        self.window.blit(episodes_text, (self.window.get_width() - episodes_text_width, 0))
        # 更新窗口显示内容
        pygame.display.update()
        self.clock.tick(self.fps)

    def iseat(self,snake):
        # 判断蛇是否吃到食物，如果是，执行以下操作
        if snake.x == self.food_.x and snake.y == self.food_.y:
            self.sound.play()
            food_born = False
            while not food_born:
                # 随机生成一个新的食物位置
                self.food_.x = random.randint(0, self.size - 1) * 40
                self.food_.y = random.randint(0, self.size - 1) * 40
                # 判断新生成的食物位置是否与蛇的身体重合，如果不重合，退出循环
                num = 0
                for segment in self.snake_body:
                    if segment.x == self.food_.x and segment.y == self.food_.y:
                        num += 1
                        continue
                if num == 0:
                    food_born = True
            # 增加蛇的长度
            self.snake_length += 1
            # 增加分数
            self.score += 10


    def reset(self):
        self.epoch += 1
        # 设置蛇和食物的大小
        self.snake_size = 40
        self.food_size = 40
        # 设置蛇的初始位置和方向
        self.snake_x = self.size // 2 * 40
        self.snake_y = self.size // 2 * 40
        self.snake_dx = 0
        self.snake_dy = 0
        # 设置蛇的初始长度和身体列表
        self.snake_length = 2
        self.snake_body = []
        snake = Snake(self.snake_x, self.snake_y,self.snake_size)
        self.snake_body.append(snake)
        snake2 = Snake(self.snake_x - self.snake_size, self.snake_y,self.snake_size)
        self.snake_body.append(snake2)
        # 设置食物的初始位置
        self.food_x = random.randint(0, self.size-1) * 40
        self.food_y = random.randint(0, self.size-1) * 40
        # 设置游戏的初始状态和分数
        self.game_over = False
        self.score = 0
        # 创建一个食物对象，传入食物的位置，颜色和大小参数
        self.food_ = Food(self.food_x, self.food_y, self.red, self.food_size)
        self.last_action = 4
        self.len_num = 0
        self.step_num = 0
        return self.obs_()

    def obs_(self):
        #蛇头和食物的相对x坐标和y坐标，蛇头上、下、左、右是否有自身身体或者游戏边界作为state,并放在1*10tensor中
        if self.device == 'cuda':
            obs = zeros(1,self.n_obs).to(self.device)
        else:
            obs = zeros(1,self.n_obs)
        #蛇头和食物的相对x坐标和y坐标
        obs[0][0] = (self.snake_body[0].x - self.food_.x)/40
        obs[0][1] = (self.snake_body[0].x - self.food_.y)/40
        #蛇头上、下、左、右是否有自身身体，如果有有多远
        for segment in self.snake_body[1:]:
            #上
            if self.snake_body[0].x == segment.x and self.snake_body[0].y - segment.y > 0:
                obs[0][2] = (self.snake_body[0].y - segment.y)/40 - 1
            else:
                obs[0][2] = -1
            #下
            if self.snake_body[0].x == segment.x and self.snake_body[0].y - segment.y < 0:
                obs[0][3] = (segment.y - self.snake_body[0].y)/40 - 1
            else:
                obs[0][3] = -1
            #左
            if self.snake_body[0].y == segment.y and self.snake_body[0].x - segment.x > 0:
                obs[0][4] = (self.snake_body[0].x - segment.x)/40 - 1
            else:
                obs[0][4] = -1
            #右
            if self.snake_body[0].y == segment.y and self.snake_body[0].x - segment.x < 0:
                obs[0][5] = (segment.x - self.snake_body[0].x)/40 - 1
            else:
                obs[0][5] = -1

        #蛇头的方向
        if self.snake_dx == 0 and self.snake_dy < 0:
            obs[0][6] = 1
        elif self.snake_dx == 0 and self.snake_dy > 0:
            obs[0][6] = 2
        elif self.snake_dx < 0 and self.snake_dy == 0:
            obs[0][6] = 3
        elif self.snake_dx > 0 and self.snake_dy == 0:
            obs[0][6] = 4
        #蛇头是否在边界
        obs[0][7] = self.snake_x == 0 or self.snake_x == 360 or self.snake_y == 0 or self.snake_y == 360
        #蛇头距离四个边界的距离
        obs[0][8] = self.snake_body[0].x/40
        obs[0][9] = self.snake_body[0].y/40
        obs[0][10] = (self.size*40-40-self.snake_body[0].x)/40
        obs[0][11] = (self.size*40-40-self.snake_body[0].y)/40
        #食物是否在边界
        obs[0][12] = self.food_.x == 0 or self.food_.x == 360 or self.food_.y == 0 or self.food_.y == 360
        #食物距离四个边界的距离
        obs[0][13] = self.food_.x/40
        obs[0][14] = self.food_.y/40
        obs[0][15] = (self.size*40-40-self.food_.x)/40
        obs[0][16] = (self.size*40-40-self.food_.y)/40
        #蛇头与食物之间是否有障碍物
        if self.snake_body[0].x == self.food_.x:
            for segment in self.snake_body[1:]:
                if segment.y > min(self.snake_body[0].y,self.food_.y) and segment.y < max(self.snake_body[0].y,self.food_.y):
                    obs[0][17] = 1
                    break
        elif self.snake_body[0].y == self.food_.y:
            for segment in self.snake_body[1:]:
                if segment.x > min(self.snake_body[0].x,self.food_.x) and segment.x < max(self.snake_body[0].x,self.food_.x):
                    obs[0][17] = 1
                    break
        else:
            obs[0][17] = 0
        return obs


class DuelingDQN(nn.Module):
    def __init__(self, n_act,mid_dim=512):
        super(DuelingDQN, self).__init__()
        self.relu = nn.LeakyReLU()
        # Flatten followed by common dense layer
        self.fc = nn.Linear(18, mid_dim)
        self.dropout = nn.Dropout(0.2)
        # Value stream
        self.value_fc1 = nn.Linear(mid_dim, mid_dim)
        self.value_fc2 = nn.Linear(mid_dim, mid_dim)
        self.value_fc3 = nn.Linear(mid_dim, mid_dim)
        self.value_fc4 = nn.Linear(mid_dim, mid_dim)
        self.value_fc5 = nn.Linear(mid_dim, mid_dim)  # This outputs a single value: V(s)
        self.value = nn.Linear(mid_dim, 1)  # This outputs a single value: V(s)


        # Advantage stream
        self.advantage_fc1 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc2 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc3 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc4 = nn.Linear(mid_dim, mid_dim)
        self.advantage_fc5 = nn.Linear(mid_dim, mid_dim)
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
        advantage = self.advantage(advantage)  # A(s,a)

        # Combine V(s) and A(s,a) to get Q values
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals

class DQNAgent(object):

    def __init__(self, q_func,n_act,device):
        self.device = device
        print("Running on", self.device)
        self.pred_func = q_func #Q函数
        if self.device == 'cuda':
            self.pred_func.to(self.device)
        else:
            pass
        self.n_act = n_act  # 动作数量

    # 根据经验得到action
    def predict(self, obs):
        obs = obs.float()
        Q_values = self.pred_func(obs)
        action = argmax(Q_values).item()
        return action

class Test():
    def __init__(self,fps):
        self.device = device('cuda' if is_available() else 'cpu')
        sound_path = self.resource_path("eat.wav")
        self.env = game(fps=fps,device=device,sound_path=sound_path)
        self.env.fps = fps
        n_act = self.env.n_act
        if self.device == 'cuda':
            q_func = DuelingDQN(n_act).to(self.device)
        else:
            q_func = DuelingDQN(n_act)
        model_path = self.resource_path("trained_model_25000.pth")
        q_func.load_state_dict(load(model_path))
        self.agent = DQNAgent(
            q_func=q_func,
            n_act=n_act,
            device=self.device
        )

    def resource_path(self,relative_path):
        """ 获取相对于当前脚本的资源的绝对路径，无论是在PyInstaller打包的环境还是普通Python环境中。

        :param relative_path: 相对于脚本的资源路径
        :return: 资源的绝对路径
        """
        if getattr(sys, 'frozen', False):  # PyInstaller将此属性设置为True when freezing your script
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)
    def test_episode(self):
        num = 0
        obs = self.env.reset()
        while True:
            action = self.agent.predict(obs)
            next_obs, done, length = self.env.step(action)
            obs = next_obs
            num += 1
            self.env.render()
            if done: break

    def test(self):
        for epoch in range(50):
            self.test_episode()

def get_fps():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 显示输入框
    fps = simpledialog.askstring("FPS Input", "\t\t请输入你想运行的游戏帧率\n注意软件会自动适配运算设备，优先使用GPU进行计算，其次是CPU\n\t但经过测试推理过程中两者速度差距不大\n\t 如果不进行输入，帧率将设为默认的10")
    if fps == None:
        return None
    elif fps == '':
        return 10
    else:
        return int(fps)

if __name__ == '__main__':
    fps = get_fps()
    if fps != None:
        test = Test(fps)
        test.test()