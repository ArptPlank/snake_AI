---

# Snake AI

## 项目概述
本项目是一个贪吃蛇游戏的AI实现，初学强化学习，并使用DQN算法及其改进版本进行实验。

## 文件说明
- **贪吃蛇源代码**：包含两个版本贪吃蛇的全部源代码及迭代20000次后的模型。环境依赖和参数调整的细节将在下文中介绍。
- **一键运行包**：对于只想体验AI而不想配置环境的用户，可以从百度网盘下载一键运行包。注意：由于GitHub上传限制，一键运行包放置在百度网盘上。

## 一键包使用说明
下载并解压后，按照`dist -> main -> main.exe`路径找到可执行文件，双击运行。程序会自动选择计算设备，优先使用GPU，其次是CPU。

## 运行环境说明
- **非打包版本**：建议在配备NVIDIA GPU的环境下运行，确保安装有匹配的CUDA版本（项目基于CUDA 11.8构建）。
- **CUDA安装**：访问[NVIDIA CUDA 11.8 下载页面](https://developer.nvidia.com/cuda-11-8-0-download-archive)下载安装。
- **Anaconda环境**：建议使用Anaconda创建虚拟环境以避免污染主环境。具体步骤如下：
  1. 下载并安装[Anaconda](https://www.anaconda.com/)。
  2. 创建新环境：`conda create --name newName python=3.8`（将`newName`替换为环境名）。
  3. 激活环境：`conda activate newName`。
  4. 安装依赖：使用`pip install 包名`或`conda install 包名`安装需要的库。
  5. 安装PyTorch：根据你的CUDA版本，使用合适的命令安装PyTorch及相关库。

## 参数说明
详细的代码注释已包含在源代码中，具体问题可通过项目的Issue页面咨询。

## 效果展示
![Figure_1](https://github.com/ArptPlank/snake_AI/assets/128218697/e96beaec-ca31-40ff-90fa-81372b122aa0)

- 1.0版本在15000次迭代后收敛，2.0版本在约25000次迭代后收敛。
- 两个版本均采用Double DQN和Dueling DQN算法，1.0版本的价值函数和策略函数共享学习率，而2.0版本分开设置，使得2.0训练更稳定。

---
