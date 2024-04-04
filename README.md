---

# Snake AI

## 项目概述
初学强化学习，探索DQN算法及其改进版在贪吃蛇游戏中的应用。

## 文件说明
- **贪吃蛇源代码**：包括两个版本的全部源代码和迭代20000次后的模型。
- **一键运行包**：方便快捷地体验AI游戏，无需配置环境。

## 使用说明

### 一键包使用
1. 下载并解压百度网盘中的包。
2. 运行`dist -> main -> main.exe`。

### 环境设置
- 使用NVIDIA GPU。
- 安装CUDA 11.8。
- 建议在Anaconda虚拟环境下运行。

#### 创建和激活Anaconda环境
```bash
conda create --name newName python=3.8
conda activate newName
```

#### 安装依赖
```bash
pip install 包名
# 或者
conda install 包名
```

## 效果展示
![模型训练监控图](https://github.com/ArptPlank/snake_AI/assets/128218697/e96beaec-ca31-40ff-90fa-81372b122aa0)

## 版本对比
- **1.0版本**：在15000次迭代后收敛。
- **2.0版本**：在25000次迭代后收敛，更加稳定。

---
