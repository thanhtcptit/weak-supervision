import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import subprocess

from pathlib import Path
from tqdm import tqdm

from src.utils import load_json, get_basename
from src.utils.common import get_current_time_str


def main(config_path, save_dir, num_trials):
    config_name = get_basename(config_path, remove_ext=True)
    config_dict = load_json(config_path)
    accs = []
    for _ in tqdm(range(num_trials)):
        try:
            p = subprocess.run(f"python run.py eval {config_path}".split(), stdout=subprocess.PIPE)
            res = p.stdout.decode()
            print(res)

            acc = float(res.split("\n")[1])
            accs.append(acc)
        except Exception as e:
            print(e)
            continue

    if len(accs) == 0:
        accs = [0]    
    avg_acc = sum(accs) / len(accs)
    print(f"AVG Acc: {avg_acc:.4f}")
    print(f"Min Acc: {min(accs):.4f}")
    print(f"Max Acc: {max(accs):.4f}")

    if not save_dir:
        save_dir = "train_logs/bench"
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / f"result_{config_name}_{get_current_time_str()}.txt", "w") as f:
        f.write(f"Config: \n{config_dict}\n{avg_acc:.4f}\n{min(accs):.4f}\n{max(accs):.4f}")
