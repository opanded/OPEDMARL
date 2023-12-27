#!/bin/ksh

# 6 Agents
python -m maddpg_o.experiments.train_epc_select \
    --scenario=adversarial \
    --num-agents=8 \
    --num-good=4 \
    --num-adversaries=4 \
    --good-load-dir="./result/adversarial/epc/8agents" \
    --save-dir="./result/adversarial/epc/8agents" \
    --dir-ids 1 2 3\
    --num-selection=2


# 16 Agents
# python -m maddpg_o.experiments.train_epc_select \
#     --scenario=adversarial \
#     --num-agents=16 \
#     --num-good=8 \
#     --num-adversaries=8 \
#     --good-load-dir="./result/adversarial/epc/16agents" \
#     --save-dir="./result/adversarial/epc/16agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 32 Agents
# python -m maddpg_o.experiments.train_epc_select \
#     --scenario=adversarial \
#     --num-agents=32 \
#     --num-good=16 \
#     --num-adversaries=16 \
#     --good-load-dir="./result/adversarial/epc/32agents" \
#     --save-dir="./result/adversarial/epc/32agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 64 Agents
# python -m maddpg_o.experiments.train_epc_select \
#     --scenario=adversarial \
#     --num-agents=64 \
#     --num-good=32 \
#     --num-adversaries=32 \
#     --good-load-dir="./result/adversarial/epc/64agents" \
#     --save-dir="./result/adversarial/epc/64agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2
