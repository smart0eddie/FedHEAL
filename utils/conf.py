import random
import torch
import numpy as np


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

def init_path(args):
    global path_dict 
    path_dict = {
        "data_path": args.data_path,
        "base_path": args.base_path,
        "log_path": args.base_path,
        "checkpoint_path": args.checkpoint_path,
    }

def data_path() -> str:
    return path_dict["data_path"]


def base_path() -> str:
    return path_dict["base_path"]


def log_path() -> str:
    return path_dict["log_path"]


def checkpoint_path() -> str:
    return path_dict["checkpoint_path"]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
