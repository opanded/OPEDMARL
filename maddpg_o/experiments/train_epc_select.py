from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
from .train_helper.train_helpers import train
import numpy as np
import os
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

def show_group_statistics(rewards, category):
    sum_rewards = np.sum(rewards, axis=0)
    print("mean:", np.mean(sum_rewards))
    print("var:", np.var(sum_rewards))

    return np.mean(sum_rewards), np.var(sum_rewards)

def add_extra_flags(parser):
    parser = argparse.ArgumentParser(
        "多智能体环境的强化学习实验")
    # 环境
    parser.add_argument("--scenario", type=str,
                        default="grassland",
                        help="场景脚本的名称")
    parser.add_argument("--num-adversaries", type=int,
                        default=2, help="对手的数量")
    parser.add_argument("--num-good", type=int,
                        default=2, help="我方的数量")
    parser.add_argument("--num-selection", type=int,
                        default=2, help="选择的数量")
    parser.add_argument("--num-agents", type=int,
                        default=2, help="智能体的数量")
    parser.add_argument("--save-dir", type=str, default="./test/",
                        help="保存训练状态和模型的目录")
    parser.add_argument("--good-load-dir", type=str, default="./test/",
                        help="加载我方训练状态和模型的目录")
    parser.add_argument("--adv-load-dir", type=str, default="./test/",
                        help="加载对方训练状态和模型的目录")
    parser.add_argument('--dir-ids', type=int, nargs="+")
    return parser

def compete(arglist):
    import copy
    dir_ids = arglist.dir_ids
    n_competitors = len(dir_ids)
    n_good = arglist.num_good
    n_adv = arglist.num_adversaries
    n = n_good + n_adv
    k = arglist.num_selection
    results = []
    for id in dir_ids:
        print('learning id %d' %id)
        load_dir = arglist.good_load_dir + '_%d/' %id + 'report.json'
        with open(load_dir) as f:
            data = json.load(f)
        results.append(data)

    good_scores = np.zeros((n_competitors, 1 ))

    for i in range(n_competitors):
        result = results[i]
        agent_rewards = result["agent_rewards"]
        print("\n\n-- 目录中的我方智能体 {}".format(i))
        good_score, good_var = show_group_statistics(agent_rewards[n_adv:], "rewards")
        good_scores[i][0] = good_score

    avg_good_scores = np.average(good_scores, axis=-1)

    top_indices = avg_good_scores.argsort()[-k:][::-1].tolist()
    print('top_indices', top_indices)
    selection_summary = {"scenario": arglist.scenario,
                        "num agents": n,
                        "weights directory": arglist.good_load_dir,
                        "top K ids": top_indices}
    json.dump(selection_summary, open(os.path.join(arglist.save_dir, "selection_summary.json"), "w"))
    return

if __name__ == "__main__":
    arglist = parse_args(add_extra_flags)
    compete(arglist)
