# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:12:15 2019

@author: charl

Methods for plotting more than one DMD object at once
"""

import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# DMD model plotting helper functions
def plot_multiple_models(all_models,
                         truth,
                         plot_mode='reconstructed_data',
                         plot_ind=0,
                         cmap=['b', 'r', 'g', 'm', 'k']):
    """
    Plots the same field from multiple models, in two modes:
        'eigs' (eigenvalues)
        'reconstructed_data' (reconstruction of the data)
    """
    
#    fig = plt.figure()
    if plot_mode is 'reconstructed_data':
        plt.plot(truth[plot_ind], label='Truth', color='k')
        for i, m in enumerate(all_models):
            if 'dmdc' in m.label_for_plots.lower():
                this_dat = m.reconstructed_data()
            else:
                this_dat = m.reconstructed_data
            plt.plot(this_dat.real[plot_ind], label=m.label_for_plots,
                     color=cmap[i])
        plt.legend(loc='lower left')
        plt.xlabel('Time')
        
    elif plot_mode is 'eigs':
        all_models[0].plot_eig()
        legend_str = [all_models[0].label_for_plots]
        plt.plot(truth.real, truth.imag, '+k', ms=20.0)
        legend_str.append('Truth')
        for i_model in range(1, len(all_models)):
            e = all_models[i_model].eigs
            plt.plot(e.real, e.imag, 'o')
        plt.legend(legend_str, loc='lower left')
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
    
#    return fig

###############################################################################
# Grouped boxplots
# Based on: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_grouped_boxplots(data_vec, group_labels, tick_labels,
                          cmap=['r', 'b', 'g'], opt={}, show_legend=True):
#    fig = plt.figure()

    # Settings for x coordinates
    widths = 0.8/len(group_labels)
    if len(group_labels) > 1:
        offset = -1.0/2.0
        delta_offset = 2.0*abs(offset) / (len(group_labels)-1)
    else:
        offset = 0.0
        delta_offset = 0.0

    # Plot each group
    for dat, lab, c in zip(data_vec, group_labels, cmap):
        pos = np.array(range(len(dat)))*2.0 + offset
        offset += delta_offset
        this_box = plt.boxplot(dat, positions=pos, sym='', widths=widths,
                               boxprops={'linewidth':3},
                               medianprops={'linewidth':3}, **opt)
        set_box_color(this_box, c)

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=c, label=lab)
    
    if show_legend:
        plt.legend(loc='lower left')
    plt.xticks(range(0, len(tick_labels) * 2, 2), tick_labels)
    plt.xlim(-1, len(tick_labels)*2-1)
#    plt.ylim(0, 1)
    plt.tight_layout()
    
#    return fig


###############################################################################
# Circles
def get_circle(ax=None, 
               center=(0.0, 0.0), r=1.0,
               label='Unit circle', color='black'):
    this_circle = plt.Circle(
        center,
        r,
        color=color,
        fill=False,
        label='Unit circle',
        linestyle='--')
    
    return this_circle
