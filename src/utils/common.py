import os
import sys
import datetime
import subprocess

from hashlib import sha256


def get_basename(path):
    if path[-1] == "/":
        path = path[:-1]
    return os.path.basename(path)


def gen_sha256(data):
    return sha256(str(data).encode()).hexdigest()


def get_current_time_str(format="%Y%m%d_%H%M%S"):
    now = datetime.datetime.now()
    return now.strftime(format)


def multi_makedirs(dirs, exist_ok=False):
    if not isinstance(dirs, list):
        dirs = list(dirs)
    for d in dirs:
        os.makedirs(d, exist_ok=exist_ok)


def get_files_multilevel(root, pattern):
    list_files = []
    for root, _, files in os.walk(root):
        for f in files:
            if pattern in f:
                list_files.append(os.path.join(root, f))
    return list_files


def run_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    return p.communicate()
