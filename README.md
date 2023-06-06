
# Distributed multi-Agent Reinforcement Learning with One-hop Neighbors (DARL1N)

This is the code base  for implementing the DARL1N algorithm presented in the paper: [Distributed multi-Agent Reinforcement Learning with One-hop Neighbors](https://arxiv.org/abs/2202.09019) (DARL1N). This repository includes implementaions of four algorithms including DARL1N, [Evoluationary Population Curriculum](https://openreview.net/forum?id=SJxbHkrKDH) (EPC), [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1706.02275.pdf) (MADDPG) , and [Mean Field Actor Critic](https://arxiv.org/abs/1802.05438) (MFAC). The original implementaions of EPC, MFAC are contained in this [repository](https://github.com/qian18long/epciclr2020), and MADDPG is in this [repository](https://github.com/openai/maddpg).
这是用于实现论文中提出的DARL1N算法的代码库：<a href="https://arxiv.org/abs/2202.09019">分布式多代理强化学习与一跳邻居</a>（DARL1N）。这个仓库包括了四种算法的实现，包括DARL1N，<a href="https://openreview.net/forum?id=SJxbHkrKDH">进化人口课程</a>（EPC），<a href="https://arxiv.org/pdf/1706.02275.pdf">多代理深度确定性策略梯度</a>（MADDPG）和<a href="https://arxiv.org/abs/1802.05438">平均场演员评论家</a>（MFAC）。EPC，MFAC的原始实现包含在这个<a href="https://github.com/qian18long/epciclr2020">仓库</a>中，MADDPG在这个<a href="https://github.com/openai/maddpg">仓库</a>中。

## Dependancies:

- Known dependancies: python3 (3.6.9): numpy (1.19.2), gym (0.17.2), tensorflow (1.8.0), mpi4py (3.0.3), scipy (1.4.1), imageio (2.9.0), mpi4py (3.0.3); mpirun (Open MPI) (2.1.1), Ubuntu 18.04.4 LTS, ksh (sh (AT&T Research) 93u+ 2012-08-01).
- 已知的依赖项：python3 (3.6.9)：numpy (1.19.2)，gym (0.17.2)，tensorflow (1.8.0)，mpi4py (3.0.3)，scipy (1.4.1)，imageio (2.9.0)，mpi4py (3.0.3)；mpirun (Open MPI) (2.1.1)，Ubuntu 18.04.4 LTS，ksh (sh (AT&T Research) 93u+ 2012-08-01)。

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



#### Options for EPC

- `--n_cpu_per_agent`: cpu usage per agent (default: `1`)

- `--good-share-weights`: good agents share weights of the agents encoder within the model.

- `--adv-share-weights`: adversarial agents share weights of the agents encoder within the model.

- `--n-envs`: number of environments instances in parallelization.

- `--last-adv`: number of adversary agents in the last stage.

- `--last-good`: number of good agents in the last stage.

- `--good-load-dir1`: directory where training state and model of first hald of good agents are loaded from.

- `--good-load-dir2`: directory where training state and model of second hald of good agents are loaded from.

- `--timeout`: seconds to wait to get data from an empty Queue in multi-processing. If the get is not successful till the expiry of timeout seconds, an exception queue.

- `--restore`: restore training state and model from the specified load directories
(For the EPC method, you may also need to allow the system to use many processes by running the command `ulimit -n 20000` (or with a larger number) )

#### Options for DARL1N

- `--prosp-dist`: value to specify the potential neighbor, corresponding to \epsilon in the paper.
- `--num-learners`: number of learners in the distributed computing system.



## Evaluation

### Command line options:
Most options are same with training command line options. Here are other options.
- `--method`: method to use including `maddpg`, `mean_field`, `darl1n` (There is a separate script for `EPC` method).
- `--display`: displays to the screen the trained policy stored in the specified directories.


## Main files and directories desriptions:
- `.maddpg_o/experiments/train_normal.py`: train the schedules MADDPG or MFAC algorithm.

- `.maddpg_o/experiments/train_epc.py`: train the scheduled EPC algorithm.

- `.maddpg_o/experiments/train_darl1n.py`: train the scheduled DARL1N algorithm.

- `.maddpg_o/experiments/train_epc_select.py`: perform mutation and selection procedure for EPC.

- `.maddpg_o/experiments/evaluate_epc.py`: evaluation of EPC algorithm.

- `.maddpg_o/experiments/evaluate_normal.py`: evaluation of MADDPG, MFAC and EPC algorithms.

- `./maddpg_o/maddpg_local`: directory that contains helper functions for the training functions.

- `./mpe_local/multiagent/`: directory that contains code for different environments.

- `./amazon_scripts`: directory that contains scripts to coordinate the distributed computing system and run DARL1N algorithm on Amazon EC2.

- `./result`: directory that contains weights for each method in each environments.


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
