#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=ising\
    --good-sight=0.5 \
    --num-agents=9 \
    --num-adversaries=0 \
    --num-food=3 \
    --num-landmark=3 \
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --good-save-dir="./result/ising/maddpg/9agents/9agents_eva/"\
    --save-rate=100 \
    --train-rate=100 \
    --prosp-dist=0.3 \
    --max-episode-len=25 \
    --good-max-num-neighbors=9 \
    --adv-max-num-neighbors=9 \
    --method="maddpg" \
    --display \
