#!/bin/sh

# 8 agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000 \
    --num-good=4 \
    --last-adv=4 \
    --last-good=4 \
    --num-food=4 \
    --num-agents=8 \
    --num-adversaries=4 \
    --max-num-train=3000 \
    --ratio=1.5 \
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --adv-load-dir="./result/adversarial/baseline_maddpg/8agents/8agents_1" \
    --save-dir="./result/adversarial/epc/8agents/8agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --timeout=0.3 \
    --seed=16 \
    --adv-load-one-side \


# 16 agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000 \
    --num-agents=16\
    --num-good=8 \
    --last-adv=4 \
    --last-good=4\
    --num-food=8 \
    --num-adversaries=8 \
    --max-num-train=2000 \
    --ratio=2.0 \
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="./result/adversarial/epc/8agents/8agents_1" \
    --good-load-dir2="./result/adversarial/epc/8agents/8agents_1" \
    --adv-load-dir="./result/adversarial/baseline_maddpg/16agents/16agents_1" \
    --save-dir="./result/adversarial/epc/16agents/16agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --max-episode-len=30 \
    --timeout=0.3 \
    --seed=28 \
    --adv-load-one-side \
    --restore \


# 32 agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000\
    --num-good=16\
    --num-agents=32\
    --last-adv=8\
    --last-good=8\
    --num-food=16\
    --num-adversaries=16\
    --max-num-train=1000\
    --ratio=2.5\
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="./result/adversarial/epc/16agents/16agents_1" \
    --good-load-dir2="./result/adversarial/epc/16agents/16agents_1" \
    --adv-load-dir="./result/adversarial/baseline_maddpg/32agents/32agents_1" \
    --save-dir="./result/adversarial/epc/32agents/32agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --max-episode-len=35 \
    --timeout=0.3 \
    --seed=28 \
    --adv-load-one-side \
    --restore \

# 64 agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000\
    --num-good=32\
    --num-agents=64\
    --last-adv=16\
    --last-good=16\
    --num-food=32\
    --num-adversaries=32\
    --max-num-train=500\
    --ratio=3\
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="./result/adversarial/epc/32agents/32agents_1" \
    --good-load-dir2="./result/adversarial/epc/32agents/32agents_1" \
    --adv-load-dir="./result/adversarial/baseline_maddpg/64agents/64agents_1" \
    --save-dir="./result/adversarial/epc/64agents/64agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --max-episode-len=40 \
    --timeout=0.3 \
    --seed=16 \
    --adv-load-one-side \
    --restore \
