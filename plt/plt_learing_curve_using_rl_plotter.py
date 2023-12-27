from rl_plotter.logger import Logger
import pickle

exp_name = 'epc'
env_name='adversarial'
logger = Logger(exp_name=exp_name, env_name=env_name)

# total_steps
f = open(r'../result/adversarial/epc/6agents/6agents_3/train_index.pkl','rb')
episode = pickle.load(f)
print(episode)

# score
f = open(r'../result/adversarial/epc/6agents/6agents_3/good_agent.pkl','rb')
reward = pickle.load(f)
print(reward)

c=0
for i in episode:
    c+=1
    logger.update(score=[reward[c-1]],total_steps=i)



