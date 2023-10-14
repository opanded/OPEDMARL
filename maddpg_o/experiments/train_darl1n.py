import argparse
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pickle
from mpi4py import MPI
import random
import maddpg_o.maddpg_local.common.tf_util as U
from maddpg_o.maddpg_local.micro.maddpg_neighbor import MADDPGAgentTrainer
from maddpg_o.maddpg_local.micro.policy_target_policy import PolicyTrainer, PolicyTargetPolicyTrainer
import tensorflow.contrib.layers as layers
import json
import imageio
import joblib

def parse_args():
    parser = argparse.ArgumentParser("多智能体环境的强化学习实验")
    # 环境
    parser.add_argument("--scenario", type=str, default="simple", help="场景脚本的名称")
    parser.add_argument("--max-episode-len", type=int, default=25, help="最大回合长度")
    parser.add_argument("--eva-max-episode-len", type=int, default=25, help="评估最大回合长度")
    parser.add_argument("--max-num-train", type=int, default=2000, help="训练的次数")
    parser.add_argument("--num-adversaries", type=int, default=0, help="对方的数量")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="我方智能体的策略")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="对方的策略")
    # 核心训练参数
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam优化器的学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--batch-size", type=int, default=1024, help="同时优化的回合数量")
    parser.add_argument("--num-units", type=int, default=64, help="多层感知机中的单元数量")
    # 检查点
    parser.add_argument("--save-dir", type=str, default="../trained_policy/", help="保存训练状态和模型的目录")
    parser.add_argument("--save-rate", type=int, default=20, help="每完成一定数量的训练后保存模型")
    parser.add_argument("--adv-load-dir", type=str, default="", help="加载对方的训练状态和模型的目录")
    parser.add_argument("--good-load-dir", type=str, default="", help="加载我方智能体的训练状态和模型的目录")
    # 评估
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="../learning_curves/", help="保存绘图数据的目录")
    parser.add_argument("--num-good", type=int, default="0", help="我方智能体的数量")
    parser.add_argument("--num-landmarks", type=int, default="0", help="地标的数量")
    parser.add_argument("--num-agents", type=int, default="0", help="智能体的数量")
    parser.add_argument("--num-learners", type=int, default="0", help="学习者的数量")
    parser.add_argument("--last-good", type=int, default="2", help="上一阶段的我方智能体数量")
    parser.add_argument("--last-adv", type=int, default="2", help="上一阶段的对方数量")
    parser.add_argument("--good-max-num-neighbors", type=int, default="0", help="邻域中最多的我方智能体数量")
    parser.add_argument("--adv-max-num-neighbors", type=int, default="0", help="邻域中最多的对方数量")
    parser.add_argument("--num-food", type=int, default="0", help="食物的数量")
    parser.add_argument("--num-forests", type=int, default="0", help="森林的数量")
    parser.add_argument("--prosp-dist", type=float, default="0.6", help="前景邻居距离")
    parser.add_argument("--adv-sight", type=float, default="1", help="对方视野")
    parser.add_argument("--good-sight", type=float, default="1", help="我方视野")
    parser.add_argument("--ratio", type=float, default="1", help="地图的大小")
    parser.add_argument("--seed", type=int, default="1", help="随机数种子")
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--load-one-side", action="store_true", default=False)
    return parser.parse_args()


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
    if evaluate:
        from mpe_local.multiagent.environment import MultiAgentEnv
        module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
        scenario_class = importlib.import_module(module_name).Scenario
        scenario = scenario_class(n_good=arglist.num_agents - arglist.num_adversaries, n_adv=arglist.num_adversaries, n_landmarks=arglist.num_landmarks, n_food=arglist.num_food, n_forests=arglist.num_forests,
                                      no_wheel=arglist.no_wheel, good_sight=arglist.good_sight, adv_sight=arglist.adv_sight, alpha=0, ratio = arglist.ratio, max_good_neighbor = arglist.good_max_num_neighbors, max_adv_neighbor = arglist.adv_max_num_neighbors)
    else:
        from mpe_local.multiagent.environment_neighbor import MultiAgentEnv
        module_name = "mpe_local.multiagent.scenarios.{}_neighbor".format(scenario_name)
        scenario_class = importlib.import_module(module_name).Scenario
        scenario = scenario_class(n_good=arglist.num_agents - arglist.num_adversaries, n_adv=arglist.num_adversaries, n_landmarks=arglist.num_landmarks, n_food=arglist.num_food, n_forests=arglist.num_forests,
                                      no_wheel=arglist.no_wheel, good_sight=arglist.good_sight, adv_sight=arglist.adv_sight, alpha=0, ratio = arglist.ratio, prosp=arglist.prosp_dist, max_good_neighbor = arglist.good_max_num_neighbors, max_adv_neighbor = arglist.adv_max_num_neighbors)

    world = scenario.make_world()
    # 创建多智能体环境
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_agents, name, obs_shape_n, arglist, session):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_agents):
        trainers.append(trainer(
            name+"agent_%d" % i, model, obs_shape_n, session, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def evaluate_policy(evaluate_env, trainers, num_episode, display = False):
    good_episode_rewards = [0.0]
    adv_episode_rewards = [0.0]
    step = 0
    episode = 0
    frames = []
    obs_n = evaluate_env.reset()
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

        new_obs_n, rew_n, done_n, next_info_n = evaluate_env.step(action_n)
        #print(rew_n)
        for i, rew in enumerate(rew_n):
            if i < arglist.num_adversaries:
                adv_episode_rewards[-1] += rew
            else:
                good_episode_rewards[-1] += rew

        step += 1
        done = all(done_n)
        terminal = (step > (arglist.eva_max_episode_len))

        obs_n = new_obs_n
        info_n = next_info_n

        if arglist.display:
            time.sleep(0.1)
            frames.append(evaluate_env.render('rgb_array')[0])
            print('当前步是', step)
            if (terminal or done):
                imageio.mimsave('demo_num_agents_%d_%d.gif' %(arglist.num_agents, episode), frames, duration=0.15)
                frames=[]
                print('demo_num_agents_%d_%d.gif' %(arglist.num_agents, episode))

        if done or terminal:
            episode += 1
            if episode >= num_episode:
                break
            #print(good_episode_rewards[-1])
            #print(adv_episode_rewards[-1])
            good_episode_rewards.append(0)

            adv_episode_rewards.append(0)
            obs_n = evaluate_env.reset()
            step = 0

    return np.mean(good_episode_rewards), np.mean(adv_episode_rewards)


def interact_with_environments(env, trainers, node_id, steps):
    act_d = env.action_space[0].n
    for k in range(steps):
        obs_pot, neighbor = env.reset(node_id) # 邻居不包括智能体本身

        action_n = [np.zeros((act_d))] * env.n # 过渡的动作

        action_neighbor = [np.zeros((act_d))] * arglist.good_max_num_neighbors # 邻居不包括智能体本身
        target_action_neighbor = [np.zeros((act_d))] * arglist.good_max_num_neighbors

        self_action = trainers[node_id].action(obs_pot[node_id])

        action_n[node_id] = self_action
        action_neighbor[0] = self_action

        valid_neighbor = 1
        for i, obs in enumerate(obs_pot):
            if i == node_id: continue
            if len(obs) !=0 :
                #print(obs)
                other_action = trainers[i].action(obs)
                action_n[i] = other_action
                if neighbor and i in neighbor and valid_neighbor < arglist.good_max_num_neighbors:
                    action_neighbor[valid_neighbor] = other_action
                    valid_neighbor += 1
        #print(action_n)
        new_obs_neighbor, rew, done_n, next_info_n = env.step(action_n) # 邻近区域内的互动

        valid_neighbor = 1
        target_action_neighbor[0]=trainers[node_id].target_action(new_obs_neighbor[node_id])

        for k, next in enumerate(new_obs_neighbor):
            if k == node_id: continue
            if len(next) != 0 and valid_neighbor < arglist.good_max_num_neighbors:
                target_action_neighbor[valid_neighbor] = trainers[k].target_action(next)
                valid_neighbor += 1

        info_n = 0.1
        trainers[node_id].experience(obs_pot[node_id], action_neighbor, new_obs_neighbor[node_id], target_action_neighbor, rew)

    return


def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_weights(trainers, index):
    weight_file_name = os.path.join(arglist.save_dir, 'agent%d.weights' %index)
    touch_path(weight_file_name)
    weight_dict = trainers[index].get_all_weights()
    joblib.dump(weight_dict, weight_file_name)


def load_weights(trainers, index):
    # with open(arglist.save_dir + 'agent%d.weights' %i,'rb') as f:
    #     weight_dict=pickle.load(f)

    # Attention here
    if index >= arglist.num_adversaries:
        weight_dict = joblib.load(os.path.join(arglist.good_load_dir, 'agent%d.weights' %((index-arglist.num_adversaries)%arglist.last_good + arglist.last_adv)))
        trainers[index].set_all_weights(weight_dict)
    else:
        weight_dict = joblib.load(os.path.join(arglist.adv_load_dir, 'agent%d.weights' %(index%arglist.last_adv)))
        trainers[index].set_all_weights(weight_dict)


# def save_weights(trainers):
#     for i in range(len(trainers)):
#         weight_file_name = os.path.join(arglist.save_dir, 'agent%d.weights' %i)
#         touch_path(weight_file_name)
#         weight_dict = trainers[i].get_weigths()
#         joblib.dump(weight_dict, weight_file_name)
        # with open(weight_file_name, 'w+') as fp:
        #     pickle.dump(weight_dict, fp)

# def load_weights(trainers):
#     for i in range(len(trainers)):
#         # with open(arglist.save_dir + 'agent%d.weights' %i,'rb') as f:
#         #     weight_dict=pickle.load(f)
#         weight_dict = joblib.load(os.path.join(arglist.load_dir, 'agent%d.weights' %(i%arglist.last_stage_num)))
#         trainers[i].set_weigths(weight_dict)


if __name__== "__main__":
    # MPI 初始化。
    comm = MPI.COMM_WORLD
    num_node = comm.Get_size()
    node_id = comm.Get_rank()
    node_name = MPI.Get_processor_name()

    with tf.Session() as session:
        # 解析参数
        arglist = parse_args()
        seed = arglist.seed
        gamma = arglist.gamma
        num_agents = arglist.num_agents
        num_learners = arglist.num_learners # 在两侧应用中，我们只训练一侧。
        assert num_node == num_learners + 1
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)
        CENTRAL_CONTROLLER = arglist.num_adversaries
        LEARNERS = [i+CENTRAL_CONTROLLER for i in range(1, 1+num_learners)]
        env = make_env(arglist.scenario, arglist, evaluate= False)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        node_id += arglist.num_adversaries

        if (node_id == CENTRAL_CONTROLLER):
            trainers = []
            # 中央控制器只需要策略参数即可执行策略进行评估
            for i in range(num_agents):
                trainers.append(PolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
        else:
            trainers = []
            # 训练器需要 MADDPG 训练器来训练自己的智能体，而对于我方智能体只需要策略和目标策略，对于对方智能体则需要策略
            for i in range(num_agents):
                if node_id - 1 == i:
                    trainers.append(MADDPGAgentTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
                elif i >= arglist.num_adversaries: # Good agents
                    trainers.append(PolicyTargetPolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
                else: # Adversary agents
                    trainers.append(PolicyTargetPolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))


        U.initialize()
        final_good_rewards = []
        final_adv_rewards = []
        final_rewards = []
        train_time = []
        global_train_time = []
        ground_global_time = time.time()
        train_step = 0
        iter_step = 0
        num_train = 0

        if (node_id == CENTRAL_CONTROLLER):
            train_start_time = time.time()
            print('计算方案:', 'DARL1N')
            print('想定: ', arglist.scenario)
            print('智能体数量: ', num_agents)
            touch_path(arglist.save_dir)
            # if arglist.load_dir == "":
            #     arglist.load_dir = arglist.save_dir
            print('我方加载文件夹为', arglist.good_load_dir)
            print('对方加载文件夹为', arglist.adv_load_dir)
            evaluate_env = make_env(arglist.scenario, arglist, evaluate= True)

        if arglist.load_one_side:
            print('加载一侧状态...')
            # Load adversary agents weights
            one_side_weights = None
            if node_id > CENTRAL_CONTROLLER:
                load_weights(trainers, node_id - 1 - arglist.num_adversaries)
                one_side_weights = trainers[node_id - 1 - arglist.num_adversaries].get_weigths()
            one_side_weights = comm.gather(one_side_weights, root = 0)
            one_side_weights = comm.bcast(one_side_weights, root = 0)
            if node_id > CENTRAL_CONTROLLER:
                for i, agent in enumerate(trainers):
                    if i < arglist.num_adversaries:
                        agent.set_weigths(one_side_weights[i+1])


        if arglist.restore:
            # 从最后阶段加载我方智能体权重
            print('加载之前的状态...')
            touch_path(arglist.good_load_dir)
            touch_path(arglist.adv_load_dir)
            weights = None
            if node_id > CENTRAL_CONTROLLER:
                # 每个学习器中每个我方智能体的负载权重
                load_weights(trainers, node_id - 1)
                weights = trainers[node_id - 1].get_weigths()
            # 收集策略和目标策略
            weights = comm.gather(weights, root = 0)
            # 将所有智能体的策略和目标策略参数广播给每个智能体
            weights = comm.bcast(weights,root=0)

            if node_id > CENTRAL_CONTROLLER:
            # 对于每个学习者，为每个智能体设置策略和目标策略
                for i, agent in enumerate(trainers):
                    if i >= arglist.num_adversaries:
                        agent.set_weigths(weights[i+1-arglist.num_adversaries])

        comm.Barrier()
        print('开始训练...')
        start_time = time.time()
        while True:
            comm.Barrier()
            if num_train > 0:
                start_master_weights=time.time()
                weights=comm.bcast(weights,root=0)
                end_master_weights=time.time()

            if (node_id in LEARNERS):
                # Receive parameters
                if num_train == 0:
                    env_time1 = time.time()
                    interact_with_environments(env, trainers, node_id-1, 5 * arglist.batch_size)
                    env_time2 = time.time()
                    print('环境交互时间', env_time2 - env_time1)
                else:
                    for i, agent in enumerate(trainers):
                        if i >= arglist.num_adversaries:
                            agent.set_weigths(weights[i+1-arglist.num_adversaries])
                    interact_with_environments(env, trainers, node_id-1, 4 * arglist.eva_max_episode_len)

                loss = trainers[node_id-1].update(trainers)
                weights = trainers[node_id-1].get_weigths()

            if (node_id == CENTRAL_CONTROLLER):
                weights = None

            weights = comm.gather(weights, root = 0)

            if (node_id in LEARNERS):
                num_train += 1
                if num_train > arglist.max_num_train:
                    save_weights(trainers, node_id - 1)
                    break

            if(node_id == CENTRAL_CONTROLLER):
                if(num_train % arglist.save_rate == 0):
                    for i in range(num_agents):
                        if i < arglist.num_adversaries:
                            trainers[i].set_weigths(one_side_weights[i+1])
                        else:
                            trainers[i].set_weigths(weights[i+1-arglist.num_adversaries])

                    end_train_time = time.time()
                    # U.save_state(arglist.save_dir, saver=saver)
                    good_reward, adv_reward = evaluate_policy(evaluate_env, trainers, 10, display = False)
                    final_good_rewards.append(good_reward)
                    final_adv_rewards.append(adv_reward)
                    train_time.append(end_train_time - start_time)
                    print('训练迭代次数:', num_train, '我方奖励:', good_reward, '对方奖励:', adv_reward, '训练时间:', round(end_train_time - start_time, 3), '全局训练时间:', round(end_train_time- ground_global_time, 3))
                    global_train_time.append(round(end_train_time - ground_global_time, 3))
                    start_time = time.time()
                num_train += 1
                if num_train > arglist.max_num_train:
                    # save_weights(trainers)
                    good_rew_file_name = arglist.save_dir + 'good_agent.pkl'
                    with open(good_rew_file_name, 'wb') as fp:
                        pickle.dump(final_good_rewards, fp)

                    adv_rew_file_name = arglist.save_dir  + 'adv_agent.pkl'
                    with open(adv_rew_file_name, 'wb') as fp:
                        pickle.dump(final_adv_rewards, fp)

                    time_file_name = arglist.save_dir + 'train_time.pkl'
                    with open(time_file_name, 'wb') as fp:
                        pickle.dump(train_time, fp)

                    global_time_file = arglist.save_dir + 'global_time.pkl'
                    with open(global_time_file, 'wb') as fp:
                        pickle.dump(global_train_time, fp)

                    train_end_time = time.time()
                    print('总训练时间:', train_end_time - train_start_time)
                    print('平均训练时间', np.mean(train_time))
                    break
