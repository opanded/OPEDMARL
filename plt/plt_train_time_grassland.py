import pickle
# todo 输出global_time (训练时间)
# grassland
# 6agents
# f = open(r'./result/grassland/maddpg/6agents/6agents_1global_time.pkl','rb')
# data = pickle.load(f)
# print(data)
#
# f = open(r'./result/grassland/mean_field/6agents/6agents_1/global_time.pkl','rb')
# data = pickle.load(f)
# print(data)
#
# f = open(r'./result/grassland/epc/6agents/6agents_1global_time.pkl','rb')
# data = pickle.load(f)
# print(data)

# 12agents
# f = open(r'./result/grassland/maddpg/12agents/12agents_1/global_time.pkl','rb')
# data = pickle.load(f)
# print(data)
#
# f = open(r'./result/grassland/mean_field/12agents/12agents_1/global_time.pkl','rb')
# data = pickle.load(f)
# print(data)
#
# f = open(r'./result/grassland/epc/12agents/12agents_1global_time.pkl','rb')
# data = pickle.load(f)
# print(data)

# 24agents
f = open(r'./result/grassland/maddpg/24agents/24agents_1/global_time.pkl','rb')
data = pickle.load(f)
print(data)

f = open(r'./result/grassland/mean_field/24agents/24agents_1/global_time.pkl','rb')
data = pickle.load(f)
print(data)

f = open(r'./result/grassland/epc/24agents/24agents_1global_time.pkl','rb')
data = pickle.load(f)
print(data)