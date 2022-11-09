#!/bin/bash

NUM_TRIALS=${1:-50}
SAVE_DIR=${2:-""}

python run.py bench configs/imdb/snorkel-base.json -s $SAVE_DIR -n $NUM_TRIALS;
python run.py bench configs/imdb/snorkel-rand-abs.json -s $SAVE_DIR -n $NUM_TRIALS;
python run.py bench configs/imdb/snorkel-rand-dis.json -s $SAVE_DIR -n $NUM_TRIALS;
python run.py bench configs/imdb/nemo-base.json -s $SAVE_DIR -n $NUM_TRIALS;

python run.py bench configs/ytb/snorkel-base.json -s $SAVE_DIR -n $NUM_TRIALS;
python run.py bench configs/ytb/snorkel-rand-abs.json -s $SAVE_DIR -n $NUM_TRIALS;
python run.py bench configs/ytb/snorkel-rand-dis.json -s $SAVE_DIR -n $NUM_TRIALS;
python run.py bench configs/ytb/nemo-base.json -s $SAVE_DIR -n $NUM_TRIALS;