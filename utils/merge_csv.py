import os
from collections import defaultdict

import autoroot
import numpy as np

from utils.common import find_all_files, group_files
from utils.save_results import HEADER


def merge_results(results, path):
    merged = defaultdict(list)
    merged["threshold"] = results[list(results.keys())[0]][:, 0]
    for result in results.values():
        for i, k in enumerate(HEADER[1:], start=1):
            merged[k].append(result[:, i])
    for k in HEADER[1:]:
        merged[f"{k}_mean"] = np.mean(merged[k], axis=0)
        merged[f"{k}_std"] = np.std(merged[k], axis=0)
    MERGED_HEADER = ["threshold"] + [v for sublist in [[f"{k}_mean", f"{k}_std"] for k in HEADER[1:]] for v in sublist]
    np.savetxt(path, np.array([merged[k] for k in MERGED_HEADER]).T, delimiter=",", header=",".join(MERGED_HEADER))


if __name__ == "__main__":
    root_dir = autoroot.root / "logs/dolly-7B-toxicity"
    # root_dir = autoroot.root / "logs/gpt2-alpaca-toxicity"
    # root_dir = autoroot.root / "logs/gpt2-imdb-toxicity"
    pattern = "/results.csv"
    files = find_all_files(root_dir, pattern)
    print(f"find {len(files)} {pattern} in {root_dir}")
    results = {}
    for file in files:
        results[file] = np.loadtxt(file, delimiter=",")
    file_groups = group_files(results, r"(.*)/seed=")
    for group_path, file_list in file_groups.items():
        group_results = {file: results[file] for file in file_list}
        merge_results(group_results, group_path + f"/results_{len(group_results)}seeds.csv")
