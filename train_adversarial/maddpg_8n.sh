#!/bin/sh

# 8 agents
python3 -m maddpg_o.experiments.train_normal \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-adversaries=4 \
    --num-food=4 \
    --num-agents=8\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --adv-load-dir="../result/adversarial/baseline_maddpg/8agents/8agents_1/"\
    --save-dir="../result/adversarial/maddpg/8agents/8agents_1" \
    --save-rate=30 \
    --max-num-train=3000\
    --good-max-num-neighbors=8 \
    --adv-max-num-neighbors=8 \
    --ratio=1.5 \
    --seed=16 \
    --load-one-side \


# 16 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=8 \
#     --num-food=8 \
#     --num-agents=16\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/16agents/16agents_1/"\
#     --save-dir="../result/adversarial/maddpg/16agents/16agents_1/" \
#     --save-rate=30 \
#     --max-num-train=2000\
#     --good-max-num-neighbors=16 \
#     --adv-max-num-neighbors=16 \
#     --max-episode-len=30 \
#     --ratio=2 \
#     --seed=16 \
#     --load-one-side \


# 32 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=16 \
#     --num-food=16 \
#     --num-agents=32\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/32agents/32agents_1/"\
#     --save-dir="../result/adversarial/maddpg/32agents/32agents_1/" \
#     --save-rate=30 \
#     --max-num-train=1000\
#     --good-max-num-neighbors=32 \
#     --adv-max-num-neighbors=32 \
#     --max-episode-len=35 \
#     --ratio=2.5 \
#     --seed=16
#     --load-one-side \


# 64 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=32 \
#     --num-food=32 \
#     --num-agents=64\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --adv-load-dir="../result/adversarial/maddpg/baseline_maddpg/64agents/64agents_1/" \
#     --save-dir="../result/adversarial/maddpg/64agents/64agents_1/" \
#     --save-rate=30 \
#     --max-num-train=500\
#     --good-max-num-neighbors=64 \
#     --adv-max-num-neighbors=64 \
#     --max-episode-len=40 \
#     --ratio=3 \
#     --seed=16 \
#     --load-one-side \
