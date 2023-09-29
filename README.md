
# Distributed multi-Agent Reinforcement Learning with One-hop Neighbors (DARL1N)

- This is the code base  for implementing the DARL1N algorithm presented in the paper: [Distributed multi-Agent Reinforcement Learning with One-hop Neighbors](https://arxiv.org/abs/2202.09019) (DARL1N). This repository includes implementaions of four algorithms including DARL1N, [Evoluationary Population Curriculum](https://openreview.net/forum?id=SJxbHkrKDH) (EPC), [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1706.02275.pdf) (MADDPG) , and [Mean Field Actor Critic](https://arxiv.org/abs/1802.05438) (MFAC). The original implementaions of EPC, MFAC are contained in this [repository](https://github.com/qian18long/epciclr2020), and MADDPG is in this [repository](https://github.com/openai/maddpg).

- 这是用于实现论文中提出的DARL1N算法的代码库：<a href="https://arxiv.org/abs/2202.09019">分布式多代理强化学习与一跳邻居</a>（DARL1N）。这个仓库包括了四种算法的实现，包括DARL1N，<a href="https://openreview.net/forum?id=SJxbHkrKDH">进化人口课程</a>（EPC），<a href="https://arxiv.org/pdf/1706.02275.pdf">多代理深度确定性策略梯度</a>（MADDPG）和<a href="https://arxiv.org/abs/1802.05438">平均场演员评论家</a>（MFAC）。EPC，MFAC的原始实现包含在这个<a href="https://github.com/qian18long/epciclr2020">仓库</a>中，MADDPG在这个<a href="https://github.com/openai/maddpg">仓库</a>中。

## Dependancies:

错误的：
`Known dependancies: python3 (3.6.9): numpy (1.19.2), gym (0.17.2), tensorflow (1.8.0), mpi4py (3.0.3), scipy (1.4.1), imageio (2.9.0), mpi4py (3.0.3); mpirun (Open MPI) (2.1.1), Ubuntu 18.04.4 LTS, ksh (sh (AT&T Research) 93u+ 2012-08-01).`
`已知的依赖项：python3 (3.6.9)：numpy (1.19.2)，gym (0.17.2)，tensorflow (1.8.0)，mpi4py (3.0.3)，scipy (1.4.1)，imageio (2.9.0)，mpi4py (3.0.3)；mpirun (Open MPI) (2.1.1)，Ubuntu 18.04.4 LTS，ksh (sh (AT&T Research) 93u+ 2012-08-01)。`

正确的：
-         apt install mpich # 安装mpi4py==3.1.4的基础
        pip install tensorflow==1.13.1 # 最重要的核心
        pip install -e . # 项目需要的特殊依赖项
        pip install numpy==1.16.4 gym==0.13.0 mpi4py==3.1.4 protobuf==3.19.4 imageio==2.21.1 matplotlib==3.5.3 joblib==1.1.0


- The DARL1N method is developed to run in a distributed computing system consisting of multiple computation nodes. In our paper, we use Amazon EC2 to build the computing system. Instructions of running our code on the Amazon EC2 is included in the directory `amazon_scripts`. You can also run our method in a single computation node, which will become multiprocessing instead of distributed computing.
- DARL1N方法是为了在由多个计算节点组成的分布式计算系统中运行而开发的。在我们的论文中，我们使用Amazon EC2来构建计算系统。在目录 `amazon_scripts`中包含了在Amazon EC2上运行我们的代码的指令。你也可以在一个单一的计算节点上运行我们的方法，这样就会变成多进程而不是分布式计算。

- To run our code, first go to the root directory of this repository and install needed modules by `pip3 install -e .`
- 要运行我们的代码，首先进入这个仓库的根目录，然后用`pip3 install -e .`安装所需的模块。



## Quick Start

### Training
- There are four directories `train_adversarial`, `train_grassland`, `train_ising`, `train_simple_spread`, including runnable scripts for the four methods in each environment.
- 这里有四个目录  `train_adversarial `,  `train_grassland `,  `train_ising `,  `train_simple_spread `，每个环境中都包含了四种方法的可运行脚本。


### Evaluation
- There are four directories `evaluate_adversarial`, `evaluate_grassland`, `evaluate_ising`, `evaluate_simple_spread`, including runable scripts for the four methods in each environment. We provide the weights for each method in each environment with the small number of agents. You can directly run the evaluation scripts to evaluate and visualize trained agents with different methods in different environments. For Ising Model, the history of states are stored in the weight directory and needed to be plotted for visualization. Due to the file size limit of CMT system, we only provide weights for small scale settings.
- 这里有四个目录 `evaluate_adversarial`, `evaluate_grassland`, `evaluate_ising`, `evaluate_simple_spread`，每个环境中都包含了四种方法的可运行脚本。我们提供了每个环境中每种方法的权重，这些权重是在少量代理的情况下训练得到的。你可以直接运行评估脚本来评估和可视化不同方法在不同环境中训练得到的代理。对于伊辛模型，状态的历史记录存储在权重目录中，需要绘制出来进行可视化。由于CMT系统的文件大小限制，我们只提供了小规模设置的权重。


## Training

### Command-line options


#### Environment options

- `--scenario`: defines which environment to be used (options: `ising`, `simple_spread`, `grassland`, `adversarial`)
- `--scenario`：定义要使用的环境（选项： `ising`, `simple_spread`, `grassland`, `adversarial`）

- `--good-sight`: the good agent's visibility radius. (for MADDPG, MFAC and EPC, the value is set to `100`, which means the whole environment is observable, for DARL1N, this value corresponds to the neighbor distance and is set to other values smaller than the size of the environment, such as 0.2.)
- `--good-sight`：好的代理的可见半径。（对于MADDPG，MFAC和EPC，该值设置为 `100`，表示整个环境都是可观察的，对于DARL1N，该值对应于邻居距离，并设置为小于环境大小的其他值，例如0.2。）


- `--adv-sight`: the adversary agent's visibility radius, similar with the good sight.
- `--adv-sight`: 对手代理的可见半径，与好的视力类似。

- `--num-agents`: number of total agents.

- `--num-adversaries`: number of adversary agents.

- `--num-good`:number of good agents.

- `--num-food`: number of food (resources) in the scenario.

- `--max-episode-len`: maximum length of each episode for the environment.
- `--max-episode-len`： 环境中每个剧集的最大长度。

- `--ratio`: size of the environment space.
- `--ratio`: 环境空间的大小。

- `--num-episodes` total number of training iterations.
- `--num-episodes` 训练迭代的总次数。

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment.
(default: `"maddpg"` (MADDPG and DARL1N); options: {`"att-maddpg"` (EPC), `"mean-field"` (MFAC)})
- `--good-policy`: 用于环境中“好”的（非对手）策略的算法。
（默认值："maddpg"（MADDPG和DARL1N）；选项：{"att-maddpg"（EPC），"mean-field"（MFAC）}）


- `--adv-policy`: algorithm used for the adversary policies in the environment
- `--adv-policy`: 用于环境中对手策略的算法
algorithm used for the 'good' (non adversary) policies in the environment.
(default: `"maddpg"` (MADDPG and DARL1N); options: {`"att-maddpg"` (EPC), `"mean-field"` (MFAC)})
用于环境中“好”的（非对手）策略的算法。
（默认值："maddpg"（MADDPG和DARL1N）；选项：{"att-maddpg"（EPC），"mean-field"（MFAC）}）


#### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--max-num-train`: maximum number of training iterations.

- `--seed`: set training seed for reproducibility. (For the EPC method, same seed may not lead to same result because environment processes share a common buffer and collect training data asynchronously and independently. The mini-batch sampled from the buffer with the same seed may differ due to different running speed of different processes.)设置训练种子以实现可重复性。 (对于EPC方法，相同的种子可能不会导致相同的结果，因为环境进程共享一个公共缓冲区，并异步和独立地收集训练数据。从缓冲区中采样的小批量数据可能因为不同进程的运行速度不同而有所差异。)


#### Checkpointing

- `--save-dir`: directory where intermediate training results and model will be saved.保存中间训练结果和模型的目录。

- `--save-rate`: model is saved every time this number of training iterations has been completed.每完成这个数量的训练迭代，就保存一次模型。

- `--good-load-dir`: directory where training state and model of good agents are loaded from.从这个目录加载好的代理的训练状态和模型。

- `--adv-load-dir`: directory where training state and model of adversary agents are loaded from.从这个目录加载对手代理的训练状态和模型。

- `--adv-load-one-side`: load training state and model of adversary agents from the directory specified with `--adv-load-dir`.从`--adv-load-dir`指定的目录加载对手代理的训练状态和模型。



#### Options for EPC EPC方法的选项

- `--n_cpu_per_agent`: cpu usage per agent (default: `1`)每个代理的cpu使用量 (默认值: 1)

- `--good-share-weights`: good agents share weights of the agents encoder within the model.好的代理在模型中共享代理编码器的权重。

- `--adv-share-weights`: adversarial agents share weights of the agents encoder within the model.对手代理在模型中共享代理编码器的权重。

- `--n-envs`: number of environments instances in parallelization.并行化中的环境实例的数量。

- `--last-adv`: number of adversary agents in the last stage.最后阶段的对手代理的数量。

- `--last-good`: number of good agents in the last stage.最后阶段的好的代理的数量。

- `--good-load-dir1`: directory where training state and model of first hald of good agents are loaded from.从这个目录加载前半部分好的代理的训练状态和模型。

- `--good-load-dir2`: directory where training state and model of second hald of good agents are loaded from.从这个目录加载后半部分好的代理的训练状态和模型。

- `--timeout`: seconds to wait to get data from an empty Queue in multi-processing. If the get is not successful till the expiry of timeout seconds, an exception queue.在多进程中从空队列中获取数据的等待秒数。如果在超时秒数过期之前没有成功获取数据，会抛出一个异常队列。

- `--restore`: restore training state and model from the specified load directories 从指定的加载目录恢复训练状态和模型
(For the EPC method, you may also need to allow the system to use many processes by running the command `ulimit -n 20000` (or with a larger number) )(对于EPC方法，你可能还需要允许系统使用多个进程，通过运行命令`ulimit -n 20000`  (或者更大的数字) )

#### Options for DARL1N DARL1N方法的选项

- `--prosp-dist`: value to specify the potential neighbor, corresponding to \epsilon in the paper.指定潜在邻居的值，对应于论文中的\epsilon。

- `--num-learners`: number of learners in the distributed computing system.分布式计算系统中的学习者数量。



## Evaluation 评估

### Command line options: 命令行选项：
Most options are same with training command line options. Here are other options. 大多数选项和训练命令行选项相同。以下是其他选项。
- `--method`: method to use including `maddpg`, `mean_field`, `darl1n` (There is a separate script for `EPC` method).要使用的方法，包括maddpg，mean_field，darl1n (对于EPC方法有一个单独的脚本)。
- `--display`: displays to the screen the trained policy stored in the specified directories.将存储在指定目录中的训练好的策略显示到屏幕上。


## Main files and directories desriptions:主要文件和目录的描述：
- `.maddpg_o/experiments/train_normal.py`: train the schedules MADDPG or MFAC algorithm.训练预定的MADDPG或MFAC算法。

- `.maddpg_o/experiments/train_epc.py`: train the scheduled EPC algorithm.训练预定的EPC算法。

- `.maddpg_o/experiments/train_darl1n.py`: train the scheduled DARL1N algorithm.训练预定的DARL1N算法。

- `.maddpg_o/experiments/train_epc_select.py`: perform mutation and selection procedure for EPC.执行EPC的变异和选择过程。

- `.maddpg_o/experiments/evaluate_epc.py`: evaluation of EPC algorithm.评估EPC算法。

- `.maddpg_o/experiments/evaluate_normal.py`: evaluation of MADDPG, MFAC and EPC algorithms.评估MADDPG，MFAC和EPC算法。

- `./maddpg_o/maddpg_local`: directory that contains helper functions for the training functions.包含训练函数的辅助函数的目录。

- `./mpe_local/multiagent/`: directory that contains code for different environments.包含不同环境的代码的目录。

- `./amazon_scripts`: directory that contains scripts to coordinate the distributed computing system and run DARL1N algorithm on Amazon EC2.包含协调分布式计算系统和在Amazon EC2上运行DARL1N算法的脚本的目录。

- `./result`: directory that contains weights for each method in each environments.包含每种方法在每种环境中的权重的目录。

## train normal
def parse_args():
-     parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
# Environment 环境
- parser.add_argument("--scenario", type=str, default="simple", help="场景脚本的名称")
- parser.add_argument("--max-episode-len", type=int, default=25, help="最大回合长度")
- parser.add_argument("--num-episodes", type=int, default=20000, help="回合数量")
- parser.add_argument("--train-period", type=int, default=1000, help="更新参数的频率")
- parser.add_argument("--num_train", type=int, default=2000, help="训练数量")
- parser.add_argument("--num-adversaries", type=int, default=0, help="对手数量")
- parser.add_argument("--good-policy", type=str, default="maddpg", help="好的代理策略")
- parser.add_argument("--adv-policy", type=str, default="maddpg", help="对手的策略")
- parser.add_argument("--max-num-train", type=int, default=2000, help="训练数量") # training_step
# Core training parameters 核心训练参数
- parser.add_argument("--lr", type=float, default=1e-2, help="Adam 优化器的学习率")
- parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
- parser.add_argument("--batch-size", type=int, default=1024, help="同时优化的回合数量")
- parser.add_argument("--num-units", type=int, default=32, help="mlp 中的单元数量")
## Checkpointing 检查点
- parser.add_argument("--save-dir", type=str, default="./trained_policy/", help="保存训练状态和模型的目录")
- parser.add_argument("--save-rate", type=int, default=20, help="每完成这个数量的训练就保存一次模型")
- parser.add_argument("--train-rate", type=int, default=20, help="每完成这么多回合就训练一次模型")
- parser.add_argument("--adv-load-dir", type=str, default="", help="加载训练状态和模型的目录")
- parser.add_argument("--good-load-dir", type=str, default="", help="加载训练状态和模型的目录")
# Evaluation 评估
- parser.add_argument("--restore", action="store_true", default=False)
- parser.add_argument("--display", action="store_true", default=False)
- parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="保存绘图数据的目录")
- parser.add_argument("--num-good", type=int, default="0", help="好的数量")
- parser.add_argument("--num-landmarks", type=int, default="0", help="地标数量")
- parser.add_argument("--num-agents", type=int, default="0", help="代理数量")
- parser.add_argument("--num-food", type=int, default="0", help="食物数量")
- parser.add_argument("--num-forests", type=int, default="0", help="森林数量")
- parser.add_argument("--prosp-dist", type=float, default="0.6", help="预期邻居距离")
- parser.add_argument("--adv-sight", type=float, default="1", help="邻居距离")
- parser.add_argument("--good-sight", type=float, default="1", help="邻居距离")
- parser.add_argument("--ratio", type=float, default="1", help="地图大小")
- parser.add_argument("--no-wheel", action="store_true", default=False)
- parser.add_argument("--benchmark", action="store_true", default=False)
- parser.add_argument("--good-max-num-neighbors", type=int, default="0", help="邻居区域内的最大代理数量")
- parser.add_argument("--adv-max-num-neighbors", type=int, default="0", help="邻居区域内的最大代理数量")
- parser.add_argument("--seed", type=int, default="1", help="随机数的种子 & quot ; )
- parser.add_argument (& quot ; --load-one-side & quot ; , action = & quot ; store_true & quot ; , default = False )
-     return parser.parse_args()


## evaluate epc
def parse_args(add_extra_flags=None):
-    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
# Environment
-    parser.add_argument("--scenario", type=str,default="grassland",help="场景脚本的名称")
-    parser.add_argument("--map-size", type=str, default="normal",help="地图大小")
-    parser.add_argument("--good-sight", type=float, default=100,help="好的视野")
-    parser.add_argument("--adv-sight", type=float, default=100,help="对手的视野")
-    parser.add_argument("--no-wheel", action="store_true", default=False,help="不使用wheel")
-    parser.add_argument("--alpha", type=float, default=0.0,help="α参数")
-    parser.add_argument("--show-attention", action="store_true", default=False,help="显示注意力")
-    parser.add_argument("--max-episode-len", type=int,default=25, help="最大回合长度")
-    parser.add_argument("--num-episodes", type=int,default=200000, help="回合数")
-    parser.add_argument("--num-adversaries", type=int,default=2, help="对手数量")
-    parser.add_argument("--num-good", type=int,default=2, help="好的数量")
-    parser.add_argument("--num-agents", type=int,default=2, help="智能体数量")
-    parser.add_argument("--num-food", type=int,default=4, help="食物数量")
-    parser.add_argument("--good-policy", type=str,default="maddpg", help="好智能体的策略")
-    parser.add_argument("--adv-policy", type=str,default="maddpg", help="对手的策略")
-    parser.add_argument("--good-load-one-side", action="store_true", default=False,help="只加载一方的模型")
-    parser.add_argument("--adv-load-one-side", action="store_true", default=False,help="只加载一方的模型")
# Core training parameters 核心训练参数
-    parser.add_argument("--lr", type=float, default=1e-2,help="Adam 优化器的学习率")
-    parser.add_argument("--gamma", type=float,default=0.95, help="折扣因子")
-    parser.add_argument("--batch-size", type=int, default=1024,help="同时优化的回合数量")
-    parser.add_argument("--num-units", type=int, default=32,help="mlp 中的单元数量")
-    parser.add_argument("--good-num-units", type=int, help="好的单元数量")
-    parser.add_argument("--adv-num-units", type=int, help="对手的单元数量")
-    parser.add_argument("--n-cpu-per-agent", type=int, default=1, help="每个代理的 CPU 数量")
-    parser.add_argument("--good-share-weights", action="store_true", default=False, help="是否共享权重")
-    parser.add_argument("--adv-share-weights", action="store_true", default=False, help="是否共享权重")
-    parser.add_argument("--use-gpu", action="store_true", default=False, help="是否使用 GPU")
# Checkpointing
-    parser.add_argument("--good-save-dir", type=str, default="./test/",help="保存训练状态和模型的目录")
-    parser.add_argument("--adv-save-dir", type=str, default="./test/",help="保存训练状态和模型的目录")
-    parser.add_argument("--train-rate", type=int, default=100,help="每完成这么多剧集就保存一次模型")
-    parser.add_argument("--save-rate", type=int, default=1000,help="每完成这么多剧集就保存一次模型")
-    parser.add_argument("--checkpoint-rate", type=int, default=0)
-    parser.add_argument("--load-dir", type=str, default="./test/",help="加载训练状态和模型的目录")
# Evaluation
-    parser.add_argument("--restore", action="store_true", default=False)
-    parser.add_argument("--display", action="store_true", default=False)
-    parser.add_argument("--save-gif-data", action="store_true", default=False)
-    parser.add_argument("--render-gif", action="store_true", default=False)
-    parser.add_argument("--benchmark", action="store_true", default=False) 
-    parser.add_argument("--benchmark-iters", type=int, default=10000,help="基准测试的迭代次数")

-    parser.add_argument("--n-envs", type=int, default=4, help="环境数量")
-    parser.add_argument("--ratio", type=float, default=1, help="比例")
-    parser.add_argument("--save-summary", action="store_true", default=False, help="保存摘要")
-    parser.add_argument("--timeout", type=float, default=0.02, help="超时")
-    parser.add_argument("--method", type=str, default="epc", help="方法")



## Demo
### Ising model (9 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/mf.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/epc.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/darl1n.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
### Ising model (16 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/mf.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/epc16_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/darl1n.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
### Ising model (25 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/mf25.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/epc25_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/darl1n.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
 ### Ising model (64 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/mean_field_local64_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/epc64_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/darl1n64_ising_model.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
  ### Food collection (3 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/3%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
   ### Food collection (6 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/6%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Food collection (12 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/12%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Food collection (24 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/24%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
  ### Grassland (6 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/6%20agents/26.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
   ### Grassland (12 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/12%20agents/31.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Grassland (24 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/24%20agents/36.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Grassland (48 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/48%20agents/41.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Adversarial (6 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/6%20agents/26.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
   ### Adversarial (12 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/12%20agents/31.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Adversarial (24 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/24%20agents/36.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Adversarial (48 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/48%20agents/41.gif" width="270" height="200" /></td>
  </tr>
 </table>
