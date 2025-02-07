import argparse
import numpy as np
import tensorflow as tf
import time
import os
import pickle
import random
import maddpg_o.maddpg_local.common.tf_util as U
from maddpg_o.maddpg_local.micro.maddpg_normal import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import imageio
import matplotlib.pyplot as plt
from .train_helper.model_v3_test3 import   mlp_model, mean_field_adv_q_model, mean_field_agent_q_model
import joblib
from functools import partial
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser("多智能体环境的强化学习实验")
    # 环境
    parser.add_argument("--scenario", type=str, default="simple", help="场景脚本的名称")
    parser.add_argument("--max-episode-len", type=int, default=25, help="最大回合长度")
    parser.add_argument("--num-episodes", type=int, default=20000, help="回合的数量")
    parser.add_argument("--train-period", type=int, default=1000, help="更新参数的频率")
    parser.add_argument("--num_train", type=int, default=2000, help="训练的次数")
    parser.add_argument("--num-adversaries", type=int, default=0, help="对方的数量")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="我方智能体的策略")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="对方的策略")
    parser.add_argument("--max-num-train", type=int, default=2000, help="训练的次数")
    # 核心训练参数
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam优化器的学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--batch-size", type=int, default=1024, help="同时优化的回合数量")
    parser.add_argument("--num-units", type=int, default=64, help="多层感知机中的单元数量")
    # 检查点
    parser.add_argument("--save-dir", type=str, default="./trained_policy/", help="保存训练状态和模型的目录")
    parser.add_argument("--save-rate", type=int, default=20, help="每完成一定数量的训练后保存模型")
    parser.add_argument("--train-rate", type=int, default=20, help="每完成一定数量的回合后训练模型")
    parser.add_argument("--adv-load-dir", type=str, default="", help="加载训练状态和模型的目录")
    parser.add_argument("--good-load-dir", type=str, default="", help="加载训练状态和模型的目录")
    # 评估
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="保存绘图数据的目录")
    parser.add_argument("--num-good", type=int, default="0", help="我方智能体的数量")
    parser.add_argument("--num-landmarks", type=int, default="0", help="地标的数量")
    parser.add_argument("--num-agents", type=int, default="0", help="智能体的数量")
    parser.add_argument("--num-food", type=int, default="0", help="食物的数量")
    parser.add_argument("--num-forests", type=int, default="0", help="森林的数量")
    parser.add_argument("--prosp-dist", type=float, default="0.6", help="前景邻居距离")
    parser.add_argument("--adv-sight", type=float, default="1", help="对方视野")
    parser.add_argument("--good-sight", type=float, default="1", help="我方视野")
    parser.add_argument("--ratio", type=float, default="1", help="比率")
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--good-max-num-neighbors", type=int, default="0", help="邻域中最多的我方智能体数量")
    parser.add_argument("--adv-max-num-neighbors", type=int, default="0", help="邻域中最多的对方智能体数量")
    parser.add_argument("--seed", type=int, default="1", help="随机数种子")
    parser.add_argument("--load-one-side", action="store_true", default=False)
    return parser.parse_args()


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

def get_trainer(side, i, scope, env, obs_shape_n, sess):
    trainer = MADDPGAgentTrainer
    policy = arglist.adv_policy if side == "adv" else arglist.good_policy
    if policy == "maddpg":
        model_p = mlp_model
        model_q = mlp_model
    elif policy == "mean_field":
        model_p = mlp_model
        model_q = partial(mean_field_adv_q_model if side == "adv" else mean_field_agent_q_model, n_good=arglist.num_agents-arglist.num_adversaries,
                          n_adv=arglist.num_adversaries, n_land=arglist.num_food, index=i, scenario = arglist.scenario, n_act = env.action_space[0].n)
    else:
        raise NotImplementedError
    # print(obs_shape_n)
    num_units = arglist.num_units
    return trainer(scope, model_p, model_q, obs_shape_n, env.action_space, i, arglist, num_units, sess, local_q_func=False)


def get_adv_trainer(i, scope, env, obs_shape_n, sess):
    return get_trainer("adv", i, scope, env, obs_shape_n, sess)


def get_good_trainer(i, scope, env, obs_shape_n, sess):
    return get_trainer("good", i, scope, env, obs_shape_n, sess)


def interact_with_environments(env, trainers, size_transitions, train_data = True, num_episode = None):
    obs_n = env.reset()
    good_episode_rewards = [0.0]
    adv_episode_rewards = [0.0]
    step = 0
    episode = 0
    num_transitions = 0
    frames = []
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # for i in range(env.n):
        #     if np.all(np.isnan(action_n[i])):
        #         print('weights....', trainers[i].get_weigths())

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        # print('训练数据', train_data)
        # print('互动动作', action_n)
        # print('互动观测', obs_n)
        # print('交互下一个观测',new_obs_n)

        step += 1
        terminal = (step > arglist.max_episode_len)

        # 收集经验
        if (train_data):
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

        obs_n = new_obs_n
        if arglist.display:
            time.sleep(0.1)
            frames.append(env.render('rgb_array')[0])
            print('当前步', step)
            if (terminal):
                imageio.mimsave('demo_num_agents_%d_%d.gif' %(arglist.num_agents, episode), frames, duration=0.15)
                frames=[]
                print('demo_num_agents_%d_%d.gif' %(arglist.num_agents, episode))

        for i, rew in enumerate(rew_n):
            if i< arglist.num_adversaries:
                adv_episode_rewards[-1] += rew
            else:
                good_episode_rewards[-1] += rew

        if terminal:
            episode += 1
            if not train_data:
                if episode >= num_episode:
                    break
            obs_n = env.reset()
            good_episode_rewards.append(0)
            adv_episode_rewards.append(0)
            step = 0

        num_transitions += 1

        if train_data:
            if num_transitions > size_transitions:
                break


    return np.mean(good_episode_rewards), np.mean(adv_episode_rewards)

def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)

def save_weights(trainers):
    for i in range(len(trainers)):
        weight_file_name = os.path.join(arglist.save_dir, 'agent%d.weights' %i)
        touch_path(weight_file_name)
        weight_dict = trainers[i].get_weigths()
        joblib.dump(weight_dict, weight_file_name)

        # with open(weight_file_name, 'w+') as fp:
        #     pickle.dump(weight_dict, fp)

def load_weights(trainers, index):
    #for i in range(len(trainers)):
        # with open(arglist.save_dir + 'agent%d.weights' %i,'rb') as f:
        #     weight_dict=pickle.load(f)
    if index < arglist.num_adversaries:
        if arglist.load_one_side:
            only_policy = True
        else:
            only_policy = False
        weight_dict = joblib.load(os.path.join(arglist.adv_load_dir, 'agent%d.weights' %index))
        trainers[index].set_weigths(weight_dict, only_policy)
    else:
        weight_dict = joblib.load(os.path.join(arglist.good_load_dir, 'agent%d.weights' %index))
        trainers[index].set_weigths(weight_dict)



def train(arglist):
    with tf.Session() as session:
        train_start_time = time.time()
        seed = arglist.seed
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        num_agents = arglist.num_agents
        # 创造环境
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # print(env.n)
        # 创建智能体训练器
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        # num_adversaries = min(env.n, 1)
        trainers = []
        for i in range(arglist.num_adversaries):
            trainer = get_adv_trainer(i, "adv{}".format(i), env, obs_shape_n, session)
            trainers.append(trainer)

        for i in range(arglist.num_adversaries, env.n):
            trainer = get_good_trainer(i, "good{}".format(i), env, obs_shape_n, session)
            trainers.append(trainer)

        #trainers = get_trainers(env, arglist.num_adversaries, obs_shape_n, arglist)
        print('使用策略:我方策略为 {}  对方策略为 {}'.format(arglist.good_policy, arglist.adv_policy))

        # 初始化
        U.initialize()

        # Load previous results, if necessary
        # if arglist.load_dir == "":
        #     arglist.load_dir = arglist.save_dir
        touch_path(arglist.save_dir)
        if arglist.display or arglist.restore:
            print('加载之前的状态 ...')
            for i in range(num_agents):
                load_weights(trainers, i)

        if arglist.load_one_side:
            print('一侧加载状态 ...')
            for i in range(arglist.num_adversaries):
                load_weights(trainers, i)

        episode_rewards = [0.0]  # 所有智能体的奖励总和
        agent_rewards = [[0.0] for _ in range(env.n)]  # 初始化每个智能体的奖励
        final_adv_rewards = []  # 用于训练曲线的总对方智能体奖励
        final_good_rewards = []  # 用于训练曲线的总我方智能体奖励
        agent_info = [[[]]]  # 用于基准测试信息的占位符
        saver = tf.train.Saver()  # TensorFlow 模型保存器
        obs_n = env.reset()  # 重置环境并获取初始观测
        episode_step = 0  # 当前回合的步数
        train_step = 0  # 当前训练步数
        ground_global_time = time.time()  # 记录全局时间
        train_time = []  # 存储每次训练的时间
        global_train_time = []  # 存储全局训练时间
        num_train = 0  # 训练次数

        print('想定: ', arglist.scenario)
        print('智能体数量: ', arglist.num_agents)
        print('开始迭代...')

        if arglist.display or arglist.restore:
            print('显示或存储')
        else:
            env_time1 = time.time()
            interact_with_environments(env, trainers, 5 * arglist.batch_size)
            env_time2 = time.time()
            print('环境交互时间: ', env_time2 - env_time1)

        t_start = time.time()
        comp_time = 0
        if arglist.display:
            good_reward , adv_reward = interact_with_environments(env, trainers, 10 * arglist.max_episode_len, False, 5)
            print('我方的奖励是', good_reward, '对方的奖励是', adv_reward)
        else:
            while True:
                # 获得行动
                interact_with_environments(env, trainers, 4 * arglist.max_episode_len)

                # 用于显示学习到的策略
                # print('迭代', num_train)
                loss = None
                com_time_start = time.time()
                for agent in trainers:
                    agent.preupdate()

                if arglist.load_one_side:
                    for i, agent in enumerate(trainers):
                        if i >= arglist.num_adversaries: # 更新我方智能体
                            loss = agent.update(trainers)
                else:
                    for agent in trainers:
                        loss = agent.update(trainers)


                com_time_end = time.time()
                comp_time += com_time_end- com_time_start
                if (num_train % arglist.save_rate ==0):
                    comp_time = 0
                    good_reward, adv_reward = interact_with_environments(env, trainers, 10*arglist.max_episode_len, False, 5)
                    t_end = time.time()
                    print("steps: {},  我方的奖励: {}, 对方的奖励:{}, 间隔时间: {}, 全局时间: {}".format(
                        num_train, good_reward, adv_reward, round(t_end-t_start, 3), round(t_end - ground_global_time, 3)))

                    global_train_time.append(round(t_end - ground_global_time, 3))
                    train_time.append(round(t_end - t_start, 3))
                    t_start = time.time()
                    # 跟踪最后一回合的奖励
                    final_good_rewards.append(good_reward)
                    final_adv_rewards.append(adv_reward)
                num_train += 1
                # 保存最后一回合奖励以供稍后绘制训练曲线
                if num_train > arglist.max_num_train:
                    save_weights(trainers)
                    if arglist.restore:
                        good_file_name = arglist.save_dir + 'good_agent_restore.pkl'
                        adv_file_name = arglist.save_dir + 'adv_agent_restore.pkl'
                        time_file_name = arglist.save_dir + 'train_time_restore.pkl'
                        global_time_file = arglist.save_dir + 'global_time_restore.pkl'
                    else:
                        good_file_name = arglist.save_dir + 'good_agent.pkl'
                        adv_file_name = arglist.save_dir + 'adv_agent.pkl'
                        time_file_name = arglist.save_dir + 'train_time.pkl'
                        global_time_file = arglist.save_dir + 'global_time.pkl'

                    with open(good_file_name, 'wb') as fp:
                        pickle.dump(final_good_rewards, fp)

                    with open(adv_file_name, 'wb') as fp:
                        pickle.dump(final_adv_rewards, fp)

                    with open(time_file_name, 'wb') as fp:
                        pickle.dump(train_time, fp)

                    with open(global_time_file, 'wb') as fp:
                        pickle.dump(global_train_time, fp)

                    train_end_time = time.time()
                    print('总训练时间:', train_end_time - train_start_time)
                    print('平均训练时间', np.mean(train_time))
                    break

if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)
