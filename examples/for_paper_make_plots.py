"""
Train size and patch size plots from analysed ray runs.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import pandas as pd
import argparse
import numpy as np
from matplotlib.lines import Line2D
path_to_package = Path('.').absolute()

path_to_results = path_to_package/'ray_analysis_results'


def make_patch_size_plot(dataset, name_stem, model_identifiers, metric, patch_sizes: List = None, tr_sizes: List = None, avgs=False):
    """
    Make a figure with patch size of the lartpc image on the x axis and a chosen metric on the y axis.
    """
    fig, ax = plt.subplots(figsize = (12,8))
    plot_lines = []
    for model_identifier in model_identifiers:
        #hardcoded from Callum
        if model_identifier == 'neqnec':
            colour = 'blue'
            if patch_sizes is not None:
                best_metrics = [0.675, 0.714, 0.810, 0.775, 0.764]
            else:
                #train size aucs
                best_metrics = [0.675, 0.740, 0.754, 0.772, 0.803]
            linestyle = '-'
        elif model_identifier == 'shallow':
            colour = 'orange'
            if patch_sizes is not None:
                best_metrics = [0.691, 0.737, 0.735, 0.736, 0.728]
            else:
                #train size aucs
                best_metrics = [0.653, 0.705, 0.727, 0.769, 0.767]
            linestyle = '--'
        elif model_identifier == 'deep_geometric':
            colour = 'magenta'
            best_metrics =  [np.nan, np.nan, 0.838, 0.859, 0.867]
            linestyle = '--'
        elif model_identifier == 'deep':
            colour = 'green'
            best_metrics = [np.nan, np.nan, 0.817, 0.850, 0.867]
            linestyle = '--'
        
        #read from stored run analysis
        else:
            if model_identifier == 'eqnec':
                colour = 'gray'
            if model_identifier == 'eqeq':
                colour = 'purple'
            if model_identifier == 'neqnec':
                colour = 'cyan'
            linestyle = '-'
            best_metrics = []
            avg_metrics = []
            stds = []
            dataframe_name_stem = f'{dataset}_{name_stem}'
            if patch_sizes is not None:
                x_axis = patch_sizes
                dataframe_name_stem += '_patch_'
                x_axis_label = "Patch size"
            else:
                x_axis = tr_sizes
                dataframe_name_stem += '_trsize_'
                x_axis_label = "Train size"

            for x_axis_point in x_axis:
                if patch_sizes is not None:
                    dataframe_name = dataframe_name_stem+f'{x_axis_point}x{x_axis_point}_{model_identifier}'
                else:
                    dataframe_name = dataframe_name_stem+f'{x_axis_point}_{model_identifier}'
                try:
                    dataframe = pd.read_pickle(path_to_results/dataframe_name)
                    # identifying the largest metric value
                    if not avgs:
                        best_metric = dataframe[metric].sort_values(ascending=False).reset_index(drop=True)[0]
                        avg_metric = np.nan
                        std = np.nan
                    else:
                        best_10 = dataframe[metric].sort_values(ascending=False).reset_index(drop=True)[:10]
                        avg_metric = best_10.mean()
                        std = best_10.std()
                        best_metric = np.nan
                except:
                    best_metric = np.nan
                    avg_metric = np.nan
                    std = np.nan

                best_metrics.append(best_metric)
                avg_metrics.append(avg_metric)
                stds.append(std)
        if not avgs:
            line, = ax.plot(x_axis, best_metrics, marker = '.', linestyle = linestyle, label = model_identifier.upper(), linewidth = 4, markersize = 12, color = colour)
        else:
            line, = ax.plot(x_axis, avg_metrics, marker = '.', linestyle = linestyle, label = model_identifier.upper(), linewidth = 4, markersize = 12, color = colour)
            ymin = [avg - std for avg, std in zip(avg_metrics, stds)]
            ymax = [avg + std for avg, std in zip(avg_metrics, stds)]
            ax.fill_between(x_axis, ymin, ymax, color = colour, alpha = 0.5)
        plot_lines.append(line)
        ax.set_ylabel(metric.upper(), fontsize = 24, labelpad = 10)
        ax.set_xlabel(x_axis_label, fontsize = 24, labelpad = 10)
        ax.tick_params(axis = 'both', which= 'major', labelsize = 24)
        ax.grid(visible=True)

    legend1 = ax.legend(handles = plot_lines,fontsize = 20)
    

    # Dummy lines for the line style legend
    solid_line = Line2D([0], [0], color='black', linestyle='-', label='Quantum-enhanced', linewidth =4)
    dashed_line = Line2D([0], [0], color='black', linestyle='--', label='Classical', linewidth = 4)

    # Legend for the line styles
    ax.legend(handles=[solid_line, dashed_line], loc='lower right', fontsize=20)    
    ax.add_artist(legend1) 

    fig.tight_layout()
    if patch_sizes is not None:
        plot_type = "patch_size"
    else:
        plot_type = "tr_size"
    figname = f'{plot_type}_plot_{dataset}_{name_stem}_{metric}'
    if avgs:
        figname += 'avgs'
    plt.savefig(figname, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('name_stem')
    parser.add_argument('metric')
    parser.add_argument('--model_identifiers', nargs='+')
    parser.add_argument("--patch_sizes", nargs='+')
    parser.add_argument("--tr_sizes", nargs = '+')
    parser.add_argument("--avgs", action='store_true')
    parse_args = parser.parse_args()

    make_patch_size_plot(parse_args.dataset, parse_args.name_stem,
                         parse_args.model_identifiers, parse_args.metric, parse_args.patch_sizes, parse_args.tr_sizes, parse_args.avgs)
