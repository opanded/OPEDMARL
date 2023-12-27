import pickle
# rb是2进制编码文件，文本文件用r
# 每隔30步输出奖励值


# todo 输出train_index (回合数)
# simple_spread
# 12agents
# MADDPG
# f = open(r'../result/simple_spread/maddpg/12agents/12agents_1/train_index.pkl','rb')
# data = pickle.load(f)
# print(data)

# MFAC
# f = open(r'../result/simple_spread/mean_field/12agents/12agents_1/train_index.pkl','rb')
# data = pickle.load(f)
# print(data)

# EPC
# f = open(r'./result/simple_spread/epc/24agents/24agents_1train_index.pkl', 'rb')
# data = pickle.load(f)
# print(data)

# 24agents
# MADDPG
# f = open(r'../result/simple_spread/maddpg/24agents/24agents_1/train_index.pkl','rb')
# data = pickle.load(f)
# print(data)

# MFAC
# f = open(r'../result/simple_spread/mean_field/24agents/24agents_1/train_index.pkl','rb')
# data = pickle.load(f)
# print(data)

# EPC
# f = open(r'./result/simple_spread/epc/24agents/24agents_1train_index.pkl', 'rb')
# data = pickle.load(f)
# print(data)

# adversarial
# MADDPG
# f = open(r'../result/adversarial/maddpg/24agents/24agents_1/train_index.pkl','rb')
# data = pickle.load(f)
# print(data)

# MFAC
# f = open(r'../result/adversarial/mean_field/24agents/24agents_1/train_index.pkl','rb')
# data = pickle.load(f)
# print(data)

# EPC
f = open(r'../result/adversarial/epc/24agents/24agents_1train_index.pkl','rb')
data = pickle.load(f)
print(data)









