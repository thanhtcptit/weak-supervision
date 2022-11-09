#!/bin/bash

CONFIG_PATH=${1:-"configs/nemo-base.json"}
SAVE_DIR=${2:-""}

# python run.py train $CONFIG_PATH -s=$SAVE_DIR -f -v;

# python run.py train "configs/ytb/snorkel-base.json" -f -v;
# python run.py train "configs/ytb/snorkel-rand-abs.json" -f -v;
# python run.py train "configs/ytb/snorkel-rand-dis.json" -f -v;

python run.py train "configs/imdb/nemo-base.json" -f -v;
