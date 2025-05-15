import csv
import datetime
import glob
import os
import re
from collections import defaultdict
from functools import wraps

import numpy as np
import seaborn as sns
import torch
from torch import linalg as LA
from torch import nn
import argparse
import json

def compute_and_save_final_results(topic_embeddings, non_gibberish, toxicity, count, total, log_dir, device):
    def cos_dist(X, Y):
        if not isinstance(X, torch.Tensor): X = torch.stack(X)
        if not isinstance(Y, torch.Tensor): Y = torch.stack(Y)
        dot = torch.matmul(X, Y.T)
        norm_X = torch.linalg.norm(X, dim=1, keepdim=True)
        norm_Y = torch.linalg.norm(Y, dim=1, keepdim=True)
        return dot / (norm_X * norm_Y.T)

    topic_tensor = torch.tensor(topic_embeddings).to(device)
    ng_arr = np.array(non_gibberish)
    tox_arr = np.array(toxicity)

    sim_matrix = cos_dist(topic_tensor, topic_tensor)
    topk_sim, _ = torch.topk(sim_matrix, sim_matrix.shape[0], dim=1)
    diversity = (1. - topk_sim[:, 1:].mean(dim=-1)).cpu().numpy()

    F1 = 2 * diversity * tox_arr / (tox_arr + diversity + 1e-8)
    weighted_div = diversity * ng_arr
    weighted_F1 = F1 * ng_arr

    result = dict(
        topic_diversity=weighted_div.sum(),
        F1=weighted_F1.sum(),
        ASR=count / total,
        total=total,
    )
    with open(f"{log_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def str2float(x):
    try:
        return float(x)
    except:
        return x

def grad_monitor(net):
    norm = [LA.vector_norm(p.grad.detach()).square() for p in net.parameters() if p.grad is not None]
    return torch.stack(norm).sum().sqrt()


def param_monitor(net):
    norm = [LA.matrix_norm(p.detach(), 2) for p in net.parameters() if p.grad is not None and len(p.shape) > 1]
    return torch.stack(norm).max()


def split_batch(minibatch_size, batch_size, shuffle=True):
    indices = np.random.permutation(batch_size) if shuffle else np.arange(batch_size)
    for idx in range(0, batch_size, minibatch_size):
        if idx + minibatch_size * 2 >= batch_size:
            yield indices[idx:]
            break
        yield indices[idx : idx + minibatch_size]


def remove_empty_events(directory):
    regex = r"events.*"
    subdirectories = glob.glob(os.path.join(directory, "*"))

    for subdir in subdirectories:
        files = glob.glob(os.path.join(subdir, "*"))
        for file in files:
            if re.search(regex, os.path.basename(file)):
                os.remove(file)
        if os.path.isdir(subdir) and not os.listdir(subdir):
            os.rmdir(subdir)


# log
def time_remain(time, epoch, nepoch, last_epoch=0):
    time = time / (epoch - last_epoch) * (nepoch - epoch)
    output = "remain " + str(datetime.timedelta(seconds=int(time)))
    return output


class LogIt:
    def __init__(self, logfile="output.log"):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            with open(self.logfile, mode="a", encoding="utf-8") as opened_file:
                output = list(map(str, args)) if len(args) else []
                output += [f"{k}={v}" for k, v in kwargs.items()]
                opened_file.write(", ".join(output) + "\n")
            return func(*args, **kwargs)

        return wrapped_function


# net
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def disable_grad(model):
    for p in model.parameters():
        p.requires_grad = False


# files
def find_all_files(
    root_dir,
    pattern,
    suffix=None,
    prefix=None,
    return_pattern=False,
    exclude_suffix=(".png", ".txt", ".log", "config.json", ".pdf", ".yml"),
):
    file_list = []
    pattern_list = []
    if os.path.isfile(root_dir):
        m = re.search(pattern, root_dir)
        if m is not None:
            file_list.append(root_dir)
            pattern_list.append(m.groups())
    else:
        for dirname, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(exclude_suffix):
                    continue
                elif suffix and not f.endswith(suffix):
                    continue
                elif prefix and not f.startswith(prefix):
                    continue
                absolute_path = os.path.join(dirname, f)
                m = re.search(pattern, absolute_path)
                if m is not None:
                    file_list.append(absolute_path)
                    pattern_list.append(m.groups())
    if return_pattern:
        return file_list, pattern_list
    else:
        return file_list


def group_files(file_list, pattern):
    res = defaultdict(list)
    for f in file_list:
        m = re.search(pattern, f) if pattern else None
        res[m.group(1) if m else ""].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}


# plot
# from https://github.com/OrdnanceSurvey/GeoDataViz-Toolkit/blob/master/Colours/GDV%20colour%20palettes%200.7.pdf
LOOKUP = {
    "1": "#FF1F5B",
    "2": "#00CD6C",
    "3": "#009ADE",
    "4": "#AF58BA",
    "5": "#FFC61E",
    "6": "#F28522",
    "7": "#A0B1BA",
    "8": "#A6761D",
    "9": "#E9002D",
    "10": "#FFAA00",
    "11": "#00B000",
}
GDV_palettes = {
    "qualitative": {
        "6a": [1, 2, 3, 4, 5, 6],
        "5a": [1, 3, 4, 5, 6],
        "4a": [1, 3, 4, 5],
        "4b": [2, 3, 4, 5],
        "3a": [1, 3, 5],
        "3b": [2, 4, 5],
        "2a": [1, 3],
        "2b": [2, 4],
        "rag": [9, 10, 11],
    },
    "sequential": {
        "s1": ["#E4F1F7", "#C5E1EF", "#9EC9E2", "#6CB0D6", "#3C93C2", "#226E9C", "#0D4A70"],
        "s2": ["#E1F2E3", "#CDE5D2", "#9CCEA7", "#6CBA7D", "#40AD5A", "#228B3B", "#06592A"],
        "s3": ["#F9D8E6", "#F2ACCA", "#ED85B0", "#E95694", "#E32977", "#C40F5B", "#8F003B"],
        "m1": ["#B7E6A5", "#7CCBA2", "#46AEA0", "#089099", "#00718B", "#045275", "#003147"],
        "m2": ["#FCE1A4", "#FABF7B", "#F08F6E", "#E05C5C", "#D12959", "#AB1866", "#6E005F"],
        "m3": ["#FFF3B2", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"],
    },
    "diverging": {
        "d1": ["#009392", "#39B185", "#9CCB86", "#E9E29C", "#EEB479", "#E88471", "#CF597E"],
        "d2": ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"],
        "d3": ["#443F90", "#685BA7", "#A599CA", "#F5DDEB", "#F492A5", "#EA6E8A", "#D21C5E"],
        "d4": ["#008042", "#6FA253", "#B7C370", "#FCE498", "#D78287", "#BF5688", "#7C1D6F"],
    },
}

PALETTE = {
    "tab10": sns.color_palette("tab10"),
    "deep": sns.color_palette("deep"),
}
# convert GDV_palettes to PALETTE
for _, v in GDV_palettes.items():
    for kk, vv in v.items():
        if isinstance(vv[0], int):
            PALETTE[kk] = [LOOKUP[str(i)] for i in vv]
        else:
            PALETTE[kk] = vv

# from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
LINESTYLE_DICT = {
    "n_solid": "solid",
    "n_dashed": "dashed",
    "n_dotted": "dotted",
    "n_dashdot": "dashdot",
    # dashdotdotted
    "dense_dashdotdotted": (0, (3, 2, 1, 2, 1, 2)),
    # dotted
    "loose_dotted": (0, (1, 10)),
    "dotted": (0, (1, 1)),
    "dense_dotted": (0, (1, 1)),
    # dashed
    "loose_dashed": (0, (5, 10)),
    "dashed": (0, (5, 5)),
    "dense_dashed": (0, (5, 1)),
    # dashdoted
    "loose_dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "dense_dashdotted": (0, (3, 1, 1, 1)),
}
