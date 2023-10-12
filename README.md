
# Distributed multi-Agent Reinforcement Learning with One-hop Neighbors (DARL1N)

- 这是用于实现论文中提出的DARL1N算法的代码库：<a href="https://arxiv.org/abs/2202.09019">分布式一跳邻居多智能体强化学习方法</a>（DARL1N）。
-一种分布式多智能体强化学习方法，它可以处理大规模的多智能体问题，同时降低了策略和值函数的表示复杂度和训练时间。它的主要思想是将全局的智能体交互解耦，只考虑一跳邻居的信息交换。每个智能体只根据自己和一跳邻居的状态来优化自己的值函数和策略函数，这样可以大大减少学习的复杂度，同时保持了表达能力，因为每个智能体可以在不同数量和状态的邻居下进行训练。这种结构还使得DARL1N可以实现分布式训练，即每个计算节点只模拟一小部分智能体的状态转移，从而加速了大规模多智能体策略的训练过程。与现有的多智能体强化学习方法相比，DARL1N在保证策略质量的同时，显著地减少了训练时间，并且可以随着智能体数量的增加而扩展。
- 这个仓库包括了四种算法的实现，包括DARL1N，<a href="https://openreview.net/forum?id=SJxbHkrKDH">进化人口课程</a>（EPC），<a href="https://arxiv.org/pdf/1706.02275.pdf">混合环境多智能体行为-评论算法</a>（MADDPG）和<a href="https://arxiv.org/abs/1802.05438">平均场AC</a>（MFAC）。
- EPC，MFAC的原始实现包含在这个<a href="https://github.com/qian18long/epciclr2020">仓库</a>中，MADDPG在这个<a href="https://github.com/openai/maddpg">仓库</a>中。

## Dependancies:

Ubuntu18.04
-        apt install mpich # 安装mpi4py==3.1.4的基础
         pip install tensorflow==1.13.1 # 最重要的核心
         pip install -e . # 项目需要的特殊依赖项
         pip install numpy==1.16.4 gym==0.13.0 mpi4py==3.1.4 protobuf==3.19.4 imageio==2.21.1 matplotlib==3.5.3 joblib==1.1.0

DARL1N方法是为了在由多个计算节点组成的分布式计算系统中运行而开发的。在我们的论文中，我们使用Amazon EC2来构建计算系统。在目录 `amazon_scripts`中包含了在Amazon EC2上运行我们的代码的指令。你也可以在一个单一的计算节点上运行我们的方法，这样就会变成多进程而不是分布式计算。

## ImageBuild

-          docker build -t marl Docker/.
  
-          docker container run -it marl


### Training
- 这里有四个目录  `train_adversarial `,  `train_grassland `,  `train_ising `,  `train_simple_spread `，每个环境中都包含了四种方法的可运行脚本。
-          bash train_adversarial/baseline_8n.sh

### Evaluation
- 这里有四个目录 `evaluate_adversarial`, `evaluate_grassland`, `evaluate_ising`, `evaluate_simple_spread`，每个环境中都包含了四种方法的可运行脚本。我们提供了每个环境中每种方法的权重，这些权重是在少量智能体的情况下训练得到的。你可以直接运行评估脚本来评估和可视化不同方法在不同环境中训练得到的智能体。对于伊辛模型，状态的历史记录存储在权重目录中，需要绘制出来进行可视化。由于CMT系统的文件大小限制，我们只提供了小规模设置的权重。


## Training

### Command-line options


#### Environment options

- `--scenario`：定义要使用的环境（选项： `ising`, `simple_spread`, `grassland`, `adversarial`）

- `--good-sight`：好的智能体的可见半径。（对于MADDPG，MFAC和EPC，该值设置为 `100`，表示整个环境都是可观察的，对于DARL1N，该值对应于邻居距离，并设置为小于环境大小的其他值，例如0.2。）

- `--adv-sight`: 对手智能体的可见半径，与好的视力类似。

- `--num-agents`: 智能体的总数。

- `--num-adversaries`: 对手智能体数量。

- `--num-good`:我方智能体数量。

- `--num-food`: 场景中的食物(资源)数量。

- `--max-episode-len`： 环境中每个回合的最大长度。

- `--ratio`: 环境空间的大小。

- `--num-episodes` 训练迭代的总次数。

- `--good-policy`: 用于环境中“好”的（非对手）策略的算法。
（默认值："maddpg"（MADDPG和DARL1N）；选项：{"att-maddpg"（EPC），"mean-field"（MFAC）}）

- `--adv-policy`: 用于环境中对手策略的算法
（默认值："maddpg"（MADDPG和DARL1N）；选项：{"att-maddpg"（EPC），"mean-field"（MFAC）}）


#### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--max-num-train`: maximum number of training iterations.

- `--seed`: 设置训练种子以实现可重复性。 (对于EPC方法，相同的种子可能不会导致相同的结果，因为环境进程共享一个公共缓冲区，并异步和独立地收集训练数据。从缓冲区中采样的小批量数据可能因为不同进程的运行速度不同而有所差异。)


#### Checkpointing

- `--save-dir`: 保存中间训练结果和模型的目录。

- `--save-rate`: 每完成这个数量的训练迭代，就保存一次模型。

- `--good-load-dir`: 从这个目录加载好的智能体的训练状态和模型。

- `--adv-load-dir`: 从这个目录加载对手智能体的训练状态和模型。

- `--adv-load-one-side`: 从`--adv-load-dir`指定的目录加载对手智能体的训练状态和模型。



#### EPC方法的选项

- `--n_cpu_per_agent`: 每个智能体的cpu使用量 (默认值: 1)

- `--good-share-weights`: 好的智能体在模型中共享智能体编码器的权重。

- `--adv-share-weights`: 对手智能体在模型中共享智能体编码器的权重。

- `--n-envs`: 并行化中的环境实例的数量。

- `--last-adv`: 最后阶段的对手智能体的数量。

- `--last-good`: 最后阶段的好的智能体的数量。

- `--good-load-dir1`: 从这个目录加载前半部分好的智能体的训练状态和模型。

- `--good-load-dir2`: 从这个目录加载后半部分好的智能体的训练状态和模型。

- `--timeout`: 在多进程中从空队列中获取数据的等待秒数。如果在超时秒数过期之前没有成功获取数据，会抛出一个异常队列。

- `--restore`: 从指定的加载目录恢复训练状态和模型(对于EPC方法，你可能还需要允许系统使用多个进程，通过运行命令`ulimit -n 20000`  (或者更大的数字) )

#### DARL1N方法的选项

- `--prosp-dist`: 指定潜在邻居的值，对应于论文中的\epsilon。

- `--num-learners`: 分布式计算系统中的学习者数量。



## Evaluation 评估

### 命令行选项：
大多数选项和训练命令行选项相同。以下是其他选项。
- `--method`: 要使用的方法，包括maddpg，mean_field，darl1n (对于EPC方法有一个单独的脚本)。
- `--display`: 将存储在指定目录中的训练好的策略显示到屏幕上。


## 主要文件和目录的描述：
- `.maddpg_o/experiments/train_normal.py`: 训练预定的MADDPG或MFAC算法。

- `.maddpg_o/experiments/train_epc.py`: 训练预定的EPC算法。

- `.maddpg_o/experiments/train_darl1n.py`: 训练预定的DARL1N算法。

- `.maddpg_o/experiments/train_epc_select.py`: 执行EPC的变异和选择过程。

- `.maddpg_o/experiments/evaluate_epc.py`: 评估EPC算法。

- `.maddpg_o/experiments/evaluate_normal.py`: 评估MADDPG，MFAC和EPC算法。

- `./maddpg_o/maddpg_local`: 包含训练函数的辅助函数的目录。

- `./mpe_local/multiagent/`: 包含不同环境的代码的目录。

- `./amazon_scripts`: 包含协调分布式计算系统和在Amazon EC2上运行DARL1N算法的脚本的目录。

- `./result`: 包含每种方法在每种环境中的权重的目录。

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
- parser.add_argument("--good-policy", type=str, default="maddpg", help="好的智能体策略")
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
- parser.add_argument("--num-agents", type=int, default="0", help="智能体数量")
- parser.add_argument("--num-food", type=int, default="0", help="食物数量")
- parser.add_argument("--num-forests", type=int, default="0", help="森林数量")
- parser.add_argument("--prosp-dist", type=float, default="0.6", help="预期邻居距离")
- parser.add_argument("--adv-sight", type=float, default="1", help="邻居距离")
- parser.add_argument("--good-sight", type=float, default="1", help="邻居距离")
- parser.add_argument("--ratio", type=float, default="1", help="地图大小")
- parser.add_argument("--no-wheel", action="store_true", default=False)
- parser.add_argument("--benchmark", action="store_true", default=False)
- parser.add_argument("--good-max-num-neighbors", type=int, default="0", help="邻居区域内的最大智能体数量")
- parser.add_argument("--adv-max-num-neighbors", type=int, default="0", help="邻居区域内的最大智能体数量")
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
-    parser.add_argument("--n-cpu-per-agent", type=int, default=1, help="每个智能体的 CPU 数量")
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
    <td><img src="./demos/ising/9%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/9%20agents/mf.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/9%20agents/epc.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/9%20agents/darl1n.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/ising/16%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/16%20agents/mf.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/16%20agents/epc16_ising_model.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/16%20agents/darl1n.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/ising/25%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/25%20agents/mf25.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/25%20agents/epc25_ising_model.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/25%20agents/darl1n.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/ising/64%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/64%20agents/mean_field_local64_ising_model.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/64%20agents/epc64_ising_model.gif" width="270" height="200" /></td>
    <td><img src="./demos/ising/64%20agents/darl1n64_ising_model.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/simple_spread/maddpg/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/mf/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/epc/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/darl1n/3%20agents/208.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/simple_spread/maddpg/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/mf/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/epc/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/darl1n/6%20agents/208.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/simple_spread/maddpg/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/mf/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/epc/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/darl1n/12%20agents/208.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/simple_spread/maddpg/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/mf/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/epc/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="./demos/simple_spread/darl1n/24%20agents/208.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/grassland/maddpg/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/mf/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/epc/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/darl1n/6%20agents/26.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/grassland/maddpg/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/mf/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/epc/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/darl1n/12%20agents/31.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/grassland/maddpg/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/mf/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/epc/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/darl1n/24%20agents/36.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/grassland/maddpg/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/mf/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/epc/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="./demos/grassland/darl1n/48%20agents/41.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/adversarial/maddpg/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/mf/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/epc/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/darl1n/6%20agents/26.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/adversarial/maddpg/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/mf/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/epc/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/darl1n/12%20agents/31.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/adversarial/maddpg/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/mf/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/epc/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/darl1n/24%20agents/36.gif" width="270" height="200" /></td>
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
    <td><img src="./demos/adversarial/maddpg/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/mf/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/epc/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="./demos/adversarial/darl1n/48%20agents/41.gif" width="270" height="200" /></td>
  </tr>
 </table>
