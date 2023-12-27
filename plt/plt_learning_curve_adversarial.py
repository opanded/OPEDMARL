import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
# todo episode

#  adversarial
# 6agents
# episode_file = open(r'../result/adversarial/maddpg/6agents/6agents_1/train_index.pkl','rb')
# episode = pickle.load(episode_file) # list
# print(episode)

# 12agents
# episode_file = open(r'../result/adversarial/maddpg/12agents/12agents_1/train_index.pkl','rb')
# episode = pickle.load(episode_file) # list
# print(episode)

# 24agents
# episode_file = open(r'../result/adversarial/maddpg/24agents/24agents_1/train_index.pkl','rb')
# episode = pickle.load(episode_file) # list
# print(episode)

# 48agents
# episode_file = open(r'../result/adversarial/maddpg/24agents/24agents_1/train_index.pkl','rb')
# episode = pickle.load(episode_file) # list
# print(episode)

# todo reward
# adversarial
# 6agents
# MADDPG_reward_file = open(r'../result/adversarial/maddpg/6agents/6agents_1/good_agent.pkl','rb')
# MADDPG_reward = pickle.load(MADDPG_reward_file) # list
# print(MADDPG_reward)
#
# MFAC_reward_file = open(r'../result/adversarial/mean_field/6agents/6agents_1/good_agent.pkl','rb')
# MFAC_reward = pickle.load(MFAC_reward_file) # list
# print(MFAC_reward)
#
# epc_reward_file = open(r'../result/adversarial/epc/6agents/6agents_1/good_agent.pkl','rb')
# epc_reward = pickle.load(epc_reward_file) # list
# print(epc_reward)

# 12agents
# MADDPG_reward_file = open(r'../result/adversarial/maddpg/12agents/12agents_1/good_agent.pkl','rb')
# MADDPG_reward = pickle.load(MADDPG_reward_file) # list
# print(MADDPG_reward)
#
# MFAC_reward_file = open(r'../result/adversarial/mean_field/12agents/12agents_1/good_agent.pkl','rb')
# MFAC_reward = pickle.load(MFAC_reward_file) # list
# print(MFAC_reward)
#
# epc_reward_file = open(r'../result/adversarial/epc/12agents/12agents_1good_agent.pkl','rb')
# epc_reward = pickle.load(epc_reward_file) # list
# print(epc_reward)

# 24agents
MADDPG_reward_file = open(r'../result/adversarial/maddpg/24agents/24agents_1/good_agent.pkl','rb')
MADDPG_reward = pickle.load(MADDPG_reward_file) # list
print(MADDPG_reward)

MFAC_reward_file = open(r'../result/adversarial/mean_field/24agents/24agents_1/good_agent.pkl','rb')
MFAC_reward = pickle.load(MFAC_reward_file) # list
print(MFAC_reward)

epc_reward_file = open(r'../result/adversarial/epc/24agents/24agents_1good_agent.pkl','rb')
epc_reward = pickle.load(epc_reward_file) # list
print(epc_reward)

plt.plot(episode, MADDPG_reward, color='blue', label='MADDPG')
plt.plot(episode, MFAC_reward, color='green', label='MFAC')
plt.plot(episode, epc_reward, color='red', label='MARL-ES')
plt.legend(loc="center right")  # 显示图例
plt.title('learning curve')
plt.xlabel('episode')
plt.ylabel('reward')
# adversarial
# plt.savefig('./adversarial/6agents.svg',format="svg")
# plt.savefig('./adversarial/12agents.svg',format="svg")
plt.savefig('./adversarial/24agents.svg',format="svg")

plt.show()
