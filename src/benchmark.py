import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import subprocess

from pathlib import Path
from tqdm import tqdm

from src.utils import Params
from src.utils.common import get_current_time_str


def main(config_path, save_dir):
    config_dict = Params.from_file(config_path).as_dict()
    accs = []
    for _ in tqdm(range(config_dict["num_trials"])):
        try:
            p = subprocess.run(f"python run.py eval {config_path}".split(), stdout=subprocess.PIPE)
            res = p.stdout.decode()
            print(res)

            acc = float(res.split("\n")[1])
            accs.append(acc)
        except Exception:
            continue

    avg_acc = sum(accs) / len(accs)
    print(f"AVG Acc: {avg_acc:.4f}")

    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / f"result_{get_current_time_str()}.txt", "w") as f:
        f.write(f"Config: \n{config_dict}\nAVG Acc: {avg_acc:.4f}\n")
