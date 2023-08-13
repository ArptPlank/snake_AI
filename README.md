# snake_AI
初学强化学习，DQN算法及其魔改版

文件说明：
前两个文件包含了两个版本贪吃蛇的所有源代码，以及迭代20000次后的模型，如果自己电脑有配置相关环境依赖的可以下载查看代码，进行学习，当然也可以进一步调参之类的优化算法，关于环境和代码参数的说明将会在之后分别进行介绍。如果只是好奇想运行这个AI程序进行游玩且也没有相关环境的，可以直接查看一键包你的文件，你会发现只有一个文本文件，不用怀疑，确实只有一个文本文件，因为github无法上传超过25MB的文件，而这个一键包为了保证相关环境依赖完整，所以大小有点大，就放在百度网盘了，有需要的自取，当然如果你有相关环境，或者知道怎么配环境的也可以下载贪吃蛇AI一键运行版本的源代码这个要轻量级的多

一键包使用说明:
在百度网盘下载完成之后，进行解压，按照dist->main->main.exe的路径就可以找到那个可执行文件，双击运行即可，程序会自动适配选择进行运算的设备，优先使用GPU进行计算，其次是CPU，由于不涉及训练过程，只是简单推理，经过测试使用CPU和GPU进行计算的差距不大，本一键包打包的是1.0版本，但1.0版本和2.0版本在推理层面没有任何性能差距，算法也是相同的，只是在训练时的稳定性上2.0会好一点点.

运行环境说明：
本项目的非打包版本，未进行设备自动适配，且由于参数量较大（两个版本默认参数量分别是260万和310万）所以最好在有英伟达GPU的前提下下载进行训练，同时在确认有GPU后还要检查自己是否装有CUDA软件包，如果你以前没学过类似的东西电脑自带是没有的，下面是CUDA的官网链接，可以点击下载，注意本项目是基于CUDA11.8构建的，请注意选择CUDA版本https://developer.nvidia.com/cuda-11-8-0-download-archive，下载安装即可，其次本项目运行在annconda的虚拟环境下，如果你不想你的电脑主环境被搞的乌烟瘴气的话建议使用anaconda在虚拟环境里运行程序和下载依赖，这是anaconda的下载地址https://www.anaconda.com/直接下载最新版本就行，下载安装完成后打开anaconda prompt（可以在window下方搜索栏里搜），输入conda create --name newName python=3.8（建议采用3.8版本的python，Newname输入你给新环境的命名，是英文就行，如果看到下载很慢别慌conda下载慢很正常，实在不行网上有很多教程教你换源地址，问你yes or no时输入y即可），再输入conda info --envs，如果看到有你刚才创建的环境说明环境创建成功，输入conda activate Newname（你刚才取得名字），如果看到前面的(base)变成了(Newname)则说明环境已激活成功.然后就是安装所需要的库，这个时候除了pytorch外，其他所需要安装的库（你看IDE里哪个飘红了哪个就要安装），都可以用
pip install 包名 或者 conda install 包名，优先用后面这个安装，因为conda会自动检测环境兼容性，但下载很慢，如果你嫌慢的话就用前面那个，一般也没什么问题，最后再打开你的IDE将环境切换成你想要的conda环境即可.

参数说明:
代码中都有详尽的注释，直接看就行，实在看不出来的issue里面问我（主要是懒得写了）

![Figure_1](https://github.com/ArptPlank/snake_AI/assets/128218697/e96beaec-ca31-40ff-90fa-81372b122aa0)

这是1.0版本上传的模型的训练时监控的reward值
2.0版本没有因为忘了保存了

1.0版本大概在15000次收敛，2.0大概在25000次收敛

两个版本的算法均采用Double DQN和Dueling DQN的符合算法，区别仅在于1.0版本的价值函数和策略函数共享同一个学习率，而2.0版本则是分开的，初始时策略函数的学习率会更大一些，所以2.0版本训练的时候会更加稳定

对了觉得项目写的好的话记得点个star呗
