# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:17:49 2019

@author: charl

Settings for paper figures
"""

import matplotlib.pyplot as plt
import matplotlib2tikz as m2t


foldername = "C:/Users/charl/Documents/Current_work/Infinite_Series_DMD_local/figures/"

def set_plot_settings(fig):
    # Sets common settings for the paper plots
    
    all_axes = fig.axes
    font_style = {'fontname':'TimesNewRoman'}
    
    title_dict = font_style.copy()
    title_dict.update({'fontsize':16})
    
    xlabel_dict = font_style.copy()
    xlabel_dict.update({'fontsize':12})
    
    for ax in all_axes:
        ax.set_title(ax.get_title(), **title_dict)
        ax.set_xlabel(ax.get_xlabel(), **xlabel_dict)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


def save_figure(fname, fig, 
                ncols_panels=1.0, ncols_paper=1, width_per_col=8.6,
                height=5.0):
    # Saves a figure in the correct folder with the right extension
    width = '%.2fcm' % ((ncols_paper * width_per_col) / ncols_panels)
    height = '%.2fcm' % height
    plt.figure(fig)
    m2t.save(foldername + fname, figurewidth=width, figureheight=height)

def save_current_figure(fname,
                ncols_panels=1.0, ncols_paper=1, width_per_col=8.6,
                height=5.0):
    # Saves current figure
    width = '%.2fcm' % ((ncols_paper * width_per_col) / ncols_panels)
    height = '%.2fcm' % height
    m2t.save(foldername + fname + '.tex',
             figurewidth=width, figureheight=height, encoding='utf-8')
    