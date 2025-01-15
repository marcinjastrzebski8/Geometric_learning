"""
Train size and patch size plots from analysed ray runs.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import pandas as pd
import argparse
path_to_package = Path('.').absolute().parent

path_to_results = path_to_package/'ray_analysis_results'


def make_patch_size_plot(dataset, name_stem, model_identifiers, metric, patch_sizes: List):
    """
    Make a figure with patch size of the lartpc image on the x axis and a chosen metric on the y axis.
    """
    ax, fig = plt.subplots()
    for model_identifier in model_identifiers:
        best_metrics = []
        for patch_size in patch_sizes:
            dataframe_name = f'{dataset}_{name_stem}_patch_{
                patch_size}x{patch_size}_{model_identifier}'
            dataframe = pd.read_pickle(path_to_results/dataframe_name)
            # identifying the largest metric value
            best_metric = dataframe[metric].sort_values(ascending=False)[0]
            best_metrics.append(best_metric)
        ax.plot(patch_sizes, best_metrics)

    figname = f'patch_size_plot_{dataset}_{name_stem}_{metric}'
    plt.savefig(figname, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('name_stem')
    parser.add_argument('metric')
    parser.add_argument('--model_identifiers', nargs='+')
    parser.add_argument("--patch_sizes", nargs='+')
    parse_args = parser.parse_args()

    make_patch_size_plot(parse_args.dataset, parse_args.name_stem,
                         parse_args.model_identifiers, parse_args.metric, parse_args.patch_sizes)
