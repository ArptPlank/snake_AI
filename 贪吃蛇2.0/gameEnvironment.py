# 导入pygame模块
import pygame
# 导入random模块
import random
import numpy as np
import math
import torch
import torch.nn.functional as F
import collections
import sys
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
    def __init__(self):
        # 初始化pygame
        pygame.init()
        # 初始化游戏窗口尺寸
        self.size = 10
        # 设置窗口的大小和标题
        self.window_width = 40*self.size
        self.window_height = 40*self.size
        # self.window = pygame.display.set_mode((self.window_width, self.window_height))
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
        # self.snake_x = 360
        # self.snake_y = 360
        self.snake_dx = 0
        self.snake_dy = 0
        # 设置蛇的初始长度和身体列表
        self.snake_length = 2
        self.snake_body = []
        snake = Snake(self.snake_x, self.snake_y, self.snake_size)
        self.snake_body.append(snake)
        snake2 = Snake(self.snake_x-self.snake_size, self.snake_y, self.snake_size)
        self.snake_body.append(snake2)
        # 设置食物的初始位置
        self.food_x = random.randint(0, self.size-1) * 40
        self.food_y = random.randint(0, self.size-1) * 40
        # 设置游戏的初始状态和分数
        self.game_over = False
        self.score = 0
        self.isfps = False
        self.clock = pygame.time.Clock()
        # 设置字体对象和大小
        self.font = pygame.font.SysFont("arial", 32)
        # 创建一个食物对象，传入食物的位置，颜色和大小参数
        self.food_ = Food(self.food_x, self.food_y, self.red, self.food_size)
        #创建奖励
        self.reward = 0
        self.n_act = 4
        self.n_obs = 20
        self.life = 256
        self.last_action = 4
        self.distance = math.sqrt(((self.snake_body[0].x-self.food_x)/40)**2+((self.snake_body[0].y-self.food_y)/40)**2)
        self.base_distance = 3
        self.len_num = 0
        self.step_num = 0
        # 设置游戏的帧率和时钟对象
        self.fps = 20
        #self.memory = collections.deque(maxlen=30)
        #没吃到食物的惩罚
        self.punish_no_food = 0.001
        #多久没吃到食物
        self.no_food_time = 0
        #最大没吃到食物的时间
        self.no_food_time_max = 12
        #撞到自己的惩罚
        self.punish_byself = 2.2
        #撞到墙的惩罚
        self.punish_bywall = 1.7
        #吃到食物的奖励
        self.reward_byfood = 2.5
        #绕圈的惩罚
        #self.punish_byround = 0.0001
        #每走一步的惩罚或奖励比例
        self.punish_byone = 0.05
        self.reward_byone = 0.05
        #直走一定长度给予惩罚
        self.punish_step = 0.01
        self.len_num_max = 4
        #活着没死给予奖励
        self.reward_byalive = 0.01
    def step(self,action,episode):
        self.len_num += 1
        self.step_num += 1
        # action 0--上   1--下   2--左  3--右
        if action == 0:
            if self.last_action == 1:
                self.self_reward(episode)
                self.game_over = True
            self.snake_dx = 0
            self.snake_dy = -self.snake_size
            self.len_num = 0
        elif action == 1:
            if self.last_action == 0:
                self.self_reward(episode)
                self.game_over = True
            self.snake_dx = 0
            self.snake_dy = self.snake_size
            self.len_num = 0
        elif action == 2:
            if self.last_action == 3:
                self.self_reward(episode)
                self.game_over = True
            self.snake_dx = -self.snake_size
            self.snake_dy = 0
            self.len_num = 0
        elif action == 3:
            if self.last_action == 2:
                self.self_reward(episode)
                self.game_over = True
            self.snake_dx = self.snake_size
            self.snake_dy = 0
            self.len_num = 0
        self.last_action = action
        done = False
        self.life -= 1
        self.step_reward()
        self.isstraight()
        # 判断蛇是否超出窗口边界，如果是，设置游戏结束为真
        if self.snake_x+self.snake_dx < 0 or self.snake_x+self.snake_dx > self.window_width - self.snake_size or self.snake_y+self.snake_dy < 0 or self.snake_y+self.snake_dy > self.window_height - self.snake_size:
            self.wall_reward(episode) # 撞墙给予惩罚
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
                self.self_reward(episode) # 撞到自己给予惩罚
                self.game_over = True

        self.reward += self.reward_byalive

        # 判断蛇头是否与食物的位置重合
        self.iseat(snake)
        reward = self.reward
        self.reward = 0
        if self.life <= 0 or self.game_over:
            done = True
            if self.life == 0:
                reward -= (100 - self.snake_length)/10
        return self.obs_(), reward, done,self.snake_length

    def linear_gradient(self,start_color, end_color,length):
        # 计算每种颜色的增量
        r_step = (end_color[0] - start_color[0]) / 40
        g_step = (end_color[1] - start_color[1]) / 40
        b_step = (end_color[2] - start_color[2]) / 40
        # 保存每种颜色的增量
        gradient = []
        # 计算每种颜色的增量，并保存到列表中
        for i in range(length):
            r = int(start_color[0] + r_step * i)
            g = int(start_color[1] + g_step * i)
            b = int(start_color[2] + b_step * i)
            gradient.append((r, g, b))
        return gradient
    def render(self,isdraw,episodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        # 填充窗口背景颜色为黑色
        self.window.fill(self.white)
        if not isdraw:
            pygame.display.update()
            # # 设置时钟对象的延迟时间，控制游戏的帧率
            # self.clock.tick(100)
            return
        for i, segment in enumerate(self.snake_body):
            # 计算过渡颜色
            segment_color = self.linear_gradient(self.blue, self.white, self.snake_length)
            # 如果是蛇头（即列表的第一个元素），绘制一个黄色方块
            if i == 0:
                pygame.draw.rect(self.window, self.green,
                                 (segment.x, segment.y, self.snake_size, self.snake_size))
            else:
                # 如果不是蛇头，仍然绘制一个矩形
                segment.draw(self.window,segment_color[i-1])
        # 调用食物对象的绘制方法，传入窗口对象参数
        self.food_.draw(self.window)
        # 使用字体对象渲染分数文本，颜色为白色
        score_text = self.font.render("Score: " + str(self.score), True, self.black)
        # 在窗口左上角绘制分数文本
        self.window.blit(score_text, (0, 0))
        # 使用字体对象渲染帧率文本，颜色为白色
        episodes_text = self.font.render("episodes: " + str(episodes), True, self.black)
        # 获取帧率文本的大小以便正确地放在右上角
        episodes_text_width, episodes_text_height = self.font.size("episodes: " + str(episodes))
        # 在窗口右上角绘制帧率文本
        self.window.blit(episodes_text, (self.window.get_width() - episodes_text_width, 0))
        # 更新窗口显示内容
        pygame.display.update()
        # 设置时钟对象的延迟时间，控制游戏的帧率
        self.clock.tick(self.fps)

    def iseat(self,snake):
        # 判断蛇是否吃到食物，如果是，执行以下操作
        if snake.x == self.food_.x and snake.y == self.food_.y:
            self.eat_reward()
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
            self.score += 1
            self.life = 200
            self.no_food_time = 0
        else:
            self.no_food_time += 1
            self.long_time_noeat_reward()

    #吃到东西给予奖励
    def eat_reward(self):
        self.reward += self.snake_length*self.reward_byfood
        #self.reward  += self.reward_byfood

    #撞墙的惩罚
    def wall_reward(self,episodes):
        if episodes <= 4000:
            self.reward -= 5
        else:
            self.reward -= min(self.punish_bywall * (1 / self.snake_length),1)
            #self.reward -= self.punish_bywall

    #撞到自己的惩罚
    def self_reward(self,episodes):
        if episodes <= 4000:
            self.reward -= 5
        else:
            self.reward -= min(self.punish_bywall * (1 / self.snake_length),1)
            #self.reward -= self.punish_byself
    #每走一步给予一个小奖励
    def step_reward(self):
        distance = math.sqrt(((self.snake_body[0].x-self.food_x)/40)**2+((self.snake_body[0].y-self.food_y)/40)**2)
        #print(((self.distance-distance)/self.base_distance)*self.punish_byone)
        self.reward += ((self.distance-distance)/self.base_distance)*self.punish_byone
        self.distance = distance

    #太长时间没吃到东西的惩罚
    def long_time_noeat_reward(self):
        if self.no_food_time > self.no_food_time_max:
            self.reward -= self.punish_no_food
    #判断是否直走
    def isstraight(self):
        if self.len_num >= self.len_num_max:
            self.reward -= self.punish_bystep


    def reset(self):
        # 设置蛇和食物的大小
        self.snake_size = 40
        self.food_size = 40
        # 设置蛇的初始位置和方向
        self.snake_x = self.size // 2 * 40
        self.snake_y = self.size // 2 * 40
        # self.snake_x = 360
        # self.snake_y = 360
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
        # 创建奖励
        self.reward = 0
        self.life = 256
        self.last_action = 4
        self.len_num = 0
        self.step_num = 0
        return self.obs_()

    def obs_(self):
        #蛇头和食物的相对x坐标和y坐标，蛇头上、下、左、右是否有自身身体或者游戏边界作为state,并放在1*10tensor中
        obs = torch.zeros(1,self.n_obs).to(torch.device("cuda:0"))
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
                    #蛇头和这个障碍物的相对距离
                    obs[0][18] = math.fabs((self.snake_body[0].y - segment.y)/40) - 1
                    #食物和这个障碍物的相对距离
                    obs[0][19] = math.fabs((self.food_.y - segment.y)/40) - 1
                    break
        elif self.snake_body[0].y == self.food_.y:
            for segment in self.snake_body[1:]:
                if segment.x > min(self.snake_body[0].x,self.food_.x) and segment.x < max(self.snake_body[0].x,self.food_.x):
                    obs[0][17] = 1
                    #蛇头和这个障碍物的相对距离
                    obs[0][18] = math.fabs((self.snake_body[0].x - segment.x)/40) - 1
                    #食物和这个障碍物的相对距离
                    obs[0][19] = math.fabs((self.food_.x - segment.x)/40) - 1
                    break
        else:
            obs[0][17] = 0
            obs[0][18] = -1
            obs[0][19] = -1
        return obs