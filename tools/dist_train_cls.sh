#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29400}

TORCH_DISTRIBUTED_DEBUG=DETAIL

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_cls.py $CONFIG --launcher pytorch ${@:3}
