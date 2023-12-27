#!/bin/sh

# 8 agents
# mpirun -n 5 python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=adversarial \
#     --good-sight=0.2 \
#     --adv-sight=100 \
#     --num-agents=8 \
#     --num-learners=4 \
#     --num-adversaries=4 \
#     --num-food=4 \
#     --num-landmark=4\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="./result/adversarial/darl1n/8agents/8agents_1" \
#     --adv-load-dir="./result/adversarial/baseline_maddpg/8agents/8agents_1" \
#     --save-rate=30 \
#     --prosp-dist=0.1 \
#     --eva-max-episode-len=25 \
#     --good-max-num-neighbors=8 \
#     --adv-max-num-neighbors=8 \
#     --max-num-train=3000\
#     --eva-max-episode-len=25 \
#     --max-episode-len=25 \
#     --ratio=1.5 \
#     --seed=16\
#     --load-one-side \


# 16 agents
mpirun -n 7 python3  -m maddpg_o.experiments.train_darl1n \
    --scenario=adversarial \
    --good-sight=0.25 \
    --adv-sight=100 \
    --num-agents=16 \
    --num-learners=6 \
    --num-adversaries=8 \
    --num-learners=8 \
    --num-food=8 \
    --num-landmark=8\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="./result/adversarial/darl1n/16agents/16agents_1" \
    --adv-load-dir="./result/adversarial/baseline_maddpg/16agents/16agents_1" \
    --save-rate=30 \
    --prosp-dist=0.15 \
    --good-max-num-neighbors=16 \
    --adv-max-num-neighbors=16 \
    --max-num-train=2000\
    --eva-max-episode-len=30 \
    --max-episode-len=30 \
    --ratio=2 \
    --seed=16\
    --batch-size=1024\
    --load-one-side \
#
#
#
#
# # 32 agents
#
mpirun -n 13 python3  -m maddpg_o.experiments.train_darl1n \
  --scenario=adversarial \
  --good-sight=0.3 \
  --adv-sight=100 \
  --num-agents=32 \
  --num-learners=12 \
  --num-adversaries=16 \
  --num-learners=16 \
  --num-food=16 \
  --num-landmark=16\
  --good-policy=maddpg \
  --adv-policy=maddpg \
  --save-dir="./result/adversarial/darl1n/32agents/32agents_1" \
  --adv-load-dir="./result/adversarial/baseline_maddpg/32agents/32agents_1" \
  --save-rate=30 \
  --prosp-dist=0.2 \
  --eva-max-episode-len=35 \
  --max-episode-len=35 \
  --good-max-num-neighbors=32 \
  --adv-max-num-neighbors=32 \
  --max-num-train=1000\
  --ratio=2.5 \
  --seed=16\
  --batch-size=1024\
  --load-one-side \
#
#
#
#
#

# 64 agents
mpirun -n 25 python3  -m maddpg_o.experiments.train_darl1n \
    --scenario=adversarial \
    --good-sight=0.35 \
    --adv-sight=100 \
    --num-agents=64 \
    --num-learners=24 \
    --num-adversaries=32 \
    --num-food=32 \
    --num-landmark=32\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="./result/adversarial/darl1n/64agents/64agents_1" \
    --adv-load-dir="./result/adversarial/baseline_maddpg/64agents/64agents_1" \
    --save-rate=30 \
    --prosp-dist=0.05 \
    --eva-max-episode-len=40 \
    --max-episode-len=40 \
    --good-max-num-neighbors=64 \
    --adv-max-num-neighbors=64 \
    --max-num-train=500\
    --eva-max-episode-len=40 \
    --max-episode-len=40 \
    --ratio=3 \
    --seed=16\
    --load-one-side \
