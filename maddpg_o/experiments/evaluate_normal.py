import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
from mpi4py import MPI
import random
import maddpg_o.maddpg_local.common.tf_util as U
from .train_helper.model_v3_test3 import   mlp_model, mean_field_adv_q_model, mean_field_agent_q_model
import tensorflow.contrib.layers as layers
import json
import imageio
import matplotlib.pyplot as plt
from functools import partial
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser("多智能体环境的强化学习实验")
    # 环境
    parser.add_argument("--scenario", type=str, default="simple", help="场景脚本的名称")
    parser.add_argument("--max-episode-len", type=int, default=25, help="最大回合长度")
    parser.add_argument("--num-episodes", type=int, default=20000, help="回合的数量")
    parser.add_argument("--train-period", type=int, default=1000, help="更新参数的频率")
    parser.add_argument("--num_train", type=int, default=2000, help="训练次数")
    parser.add_argument("--num-adversaries", type=int, default=0, help="对手的数量")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="我方代理的策略")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="对手的策略")
    # 核心训练参数
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam优化器的学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--batch-size", type=int, default=1024, help="同时优化的回合数量")
    parser.add_argument("--num-units", type=int, default=64, help="多层感知机中的单元数量")
    # 检查点
    parser.add_argument("--good-save-dir", type=str, default="./trained_policy/", help="保存我方训练状态和模型的目录")
    parser.add_argument("--adv-save-dir", type=str, default="./trained_policy/", help="保存对手训练状态和模型的目录")
    parser.add_argument("--save-rate", type=int, default=20, help="每完成一定数量的训练后保存模型")
    parser.add_argument("--train-rate", type=int, default=20, help="每完成一定数量的回合后训练模型")
    parser.add_argument("--load-dir", type=str, default="", help="加载训练状态和模型的目录")
    # 评估
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="保存绘图数据的目录")
    parser.add_argument("--num-good", type=int, default="0", help="我方代理的数量")
    parser.add_argument("--num-landmarks", type=int, default="0", help="地标的数量")
    parser.add_argument("--num-agents", type=int, default="0", help="智能体的数量")
    parser.add_argument("--last-stage-num", type=int, default="0", help="来自上一阶段的智能体数量")
    parser.add_argument("--good-max-num-neighbors", type=int, default="0", help="邻域中最多的我方代理数量")
    parser.add_argument("--adv-max-num-neighbors", type=int, default="0", help="邻域中最多的对手数量")
    parser.add_argument("--num-food", type=int, default="0", help="食物的数量")
    parser.add_argument("--num-forests", type=int, default="0", help="森林的数量")
    parser.add_argument("--prosp-dist", type=float, default="0.6", help="前景邻居距离")
    parser.add_argument("--adv-sight", type=float, default="1", help="邻居距离")
    parser.add_argument("--good-sight", type=float, default="1", help="邻居距离")
    parser.add_argument("--ratio", type=float, default="1", help="地图大小")
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="darl1n")
    return parser.parse_args()

def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # 该模型将观察结果作为输入并返回所有操作的值
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, evaluate=False): ###################
    import importlib
    from mpe_local.multiagent.environment import MultiAgentEnv
    module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
    scenario_class = importlib.import_module(module_name).Scenario
    scenario = scenario_class(n_good=arglist.num_agents - arglist.num_adversaries, n_adv=arglist.num_adversaries, n_landmarks=arglist.num_landmarks, n_food=arglist.num_food, n_forests=arglist.num_forests,
                                  no_wheel=arglist.no_wheel, good_sight=arglist.good_sight, adv_sight=arglist.adv_sight, alpha=0, ratio = arglist.ratio, max_good_neighbor = arglist.good_max_num_neighbors, max_adv_neighbor = arglist.adv_max_num_neighbors)
    world = scenario.make_world()
    # 创建多智能体环境
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_agents, name, obs_shape_n, arglist, session):
    trainers = []
    model = mlp_model
    if arglist.method == 'darl1n':
        from maddpg_o.maddpg_local.micro.policy_target_policy import PolicyTrainer
        for i in range(num_agents):
            trainers.append(PolicyTrainer(
                "actor" + "agent_%d" % i, model, obs_shape_n, session, env.action_space, i, arglist,
                False))
    else:
        from maddpg_o.maddpg_local.micro.policy_normal import PolicyTrainer
        num_units = arglist.num_units
        for i in range(arglist.num_adversaries):
            model_p = mlp_model
            if arglist.adv_policy == "mean_field":
                model_q = partial(mean_field_adv_q_model, n_good=arglist.num_agents-arglist.num_adversaries,
                                    n_adv=arglist.num_adversaries, n_land=arglist.num_food, index=i, scenario = arglist.scenario, n_act = env.action_space[0].n)
            else:
                model_q = mlp_model
            trainers.append(PolicyTrainer("adv{}".format(i), model_p, model_q, obs_shape_n, env.action_space, i, arglist, num_units, session, local_q_func=False))
        for i in range(arglist.num_adversaries, env.n):
            model_p = mlp_model
            if arglist.good_policy == "mean_field":
                model_q = partial(mean_field_agent_q_model, n_good=arglist.num_agents-arglist.num_adversaries, n_adv=arglist.num_adversaries, n_land=arglist.num_food, index=i, scenario = arglist.scenario, n_act = env.action_space[0].n)
            else:
                model_q = mlp_model
            trainers.append(PolicyTrainer("good{}".format(i), model_p, model_q, obs_shape_n, env.action_space, i, arglist, num_units, session, local_q_func=False))
    return trainers

def evaluate_policy(env, trainers, size_transitions, display = False):
    good_episode_rewards = [0.0]
    adv_episode_rewards = [0.0]
    step = 0
    num_transitions = 0
    frames = []
    initial = []
    obs_n = env.reset()
    if arglist.scenario == 'ising':
        for agent in env.world.agents:
            initial.append(agent.state.spin)
    print(initial)
    action_history = []

    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        action_history.append(action_n)

        new_obs_n, rew_n, done_n, next_info_n = env.step(action_n)
        print(rew_n)
        num_transitions+=1
        for i, rew in enumerate(rew_n):
            if i < arglist.num_adversaries:
                adv_episode_rewards[-1] += rew
            else:
                good_episode_rewards[-1] += rew
        step += 1
        done = all(done_n)
        terminal = (step > (arglist.max_episode_len))
        obs_n = new_obs_n
        info_n = next_info_n
        if display:
            if arglist.scenario == 'ising':
                if (terminal or done):
                    action_file = arglist.good_save_dir + 'history_action%d.pkl' %num_transitions
                    with open(action_file, 'wb') as fp:
                        pickle.dump(action_history, fp)
                    initial_file = arglist.good_save_dir + 'initial%d.pkl' %num_transitions
                    with open(initial_file, 'wb') as fp:
                        pickle.dump(initial, fp)
                    initial =[]
                    action_history = []
            else:
                time.sleep(0.1)
                frames.append(env.render('rgb_array')[0])
                print('The step is', step)
                if (terminal or done):
                    gif_path = './visualize/' + arglist.scenario + '/' + arglist.method + '/%dagents/gifs/' %arglist.num_agents
                    touch_path(gif_path)
                    imageio.mimsave(gif_path + '%d.gif' %num_transitions, frames, duration=0.15)
                    plt.imshow(frames[-1])
                    plt.xticks([]),plt.yticks([])
                    plt.savefig(gif_path + '%d.png' %num_transitions, transparent=True)
                    frames=[]

        if done or terminal:
            good_episode_rewards.append(0)
            adv_episode_rewards.append(0)
            obs_n  = env.reset()
            if arglist.scenario == 'ising':
                for agent in env.world.agents:
                    initial.append(agent.state.spin)
            step = 0

        if num_transitions >= size_transitions:
            print('good', good_episode_rewards, 'adv', adv_episode_rewards)
            break
        
    return np.mean(good_episode_rewards), np.mean(adv_episode_rewards)

def load_weights(trainers, index):
    if index < arglist.num_adversaries:
        weight_dict = joblib.load(os.path.join(arglist.adv_save_dir, 'agent%d.weights' %index))
    else:
        weight_dict = joblib.load(os.path.join(arglist.good_save_dir, 'agent%d.weights' %index))

    trainers[index].set_weigths(weight_dict)

if __name__== "__main__":
    #Parse the parameters
    arglist = parse_args()
    with tf.Session() as session:
        tf.set_random_seed(30)
        random.seed(30)
        np.random.seed(30)
        env = make_env(arglist.scenario, arglist, evaluate= True)
        num_agents = env.n
        print('评估计算方案: ', 'DARL1N')
        print('想定: ', arglist.scenario)
        print('智能体数量: ', num_agents)
        #touch_path(arglist.save_dir)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_agents, None, obs_shape_n, arglist, session)
        U.initialize()

        print('加载之前的状态...')
        #U.load_state(arglist.save_dir)
        for i in range(num_agents):
            load_weights(trainers, i)
        tf.set_random_seed(30)
        random.seed(30)
        np.random.seed(30)
        good_reward , bad_reward = evaluate_policy(env, trainers, 10*arglist.max_episode_len, display = arglist.display)
        print('奖励为', good_reward, bad_reward)
