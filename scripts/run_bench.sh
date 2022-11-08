#!/bin/bash

NUM_TRIALS=${1:-50}

python run.py bench configs/imdb/snorkel-base.json -n $NUM_TRIALS;
python run.py bench configs/imdb/snorkel-abs.json -n $NUM_TRIALS;
python run.py bench configs/imdb/snorkel-dis.json -n $NUM_TRIALS;
python run.py bench configs/imdb/nemo-base.json -n $NUM_TRIALS;

python run.py bench configs/ytb/snorkel-base.json -n $NUM_TRIALS;
python run.py bench configs/ytb/snorkel-abs.json -n $NUM_TRIALS;
python run.py bench configs/ytb/snorkel-dis.json -n $NUM_TRIALS;
python run.py bench configs/ytb/nemo-base.json -n $NUM_TRIALS;