# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:20:56 2019

@author: charl
"""

import test_data_utils as tdu
import test_plot_utils as tpu
from pydmd import DMDc, idmdc
import paper_settings as ps
import matplotlib.pyplot as plt
from dmd_data import DMDData
import numpy as np
import random

random.seed(1415)
#random.seed(1414)
np.random.seed(random.randint(1, 10000))
to_save = True

fig_opt = {'ncols_panels': 2.0, 'ncols_paper': 2}
diagnostic_plots = False
###############################################################################
# Compare reconstruction accuracy as a function
# of parameters (boxplots); G-D truncation
ind_var_between_plots = {'sigma': [0.01, 0.1]}
data_opt = {'n_A':4, 'n_B':1, 'dim_increase':6, 'num_datasets':25}
data_obj = DMDData(**data_opt)

if False:
    # Plot 1: DMDc as a function of rank
    all_obj = [DMDc]
    all_opt = [{'svd_rank':-1}]
    ind_var_in_plot = {'svd_rank': list(range(2,10))}
    fig = tdu.plot_reconstruction_error(all_obj, all_opt,
                                        ind_var_in_plot=ind_var_in_plot,
                                        ind_var_between_plots=ind_var_between_plots,
                                        data_opt=data_opt,
                                        error_metric='correlation',
                                        global_dat_obj=data_obj,
                                        ind_var_dat_or_model='model')
    ps.set_plot_settings(fig)
    if to_save:
        fname = 'fig_dmdc_parameters'
        plt.savefig(ps.foldername + fname, bbox_inches='tight')
        plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
        ps.save_current_figure(fname, **fig_opt)

# Plot 2: iDMD as a function of truncation
all_obj = [idmdc.iDMDc]
all_opt = [{'svd_rank':-1, 'lambda_factor':None}]

#ind_var_in_plot = {'truncation': list(range(5,80,10))}
ind_var_in_plot = {'truncation': list(range(2,10))}
#ind_var_in_plot = {'truncation': [1, 10]}
fig = tdu.plot_reconstruction_error(all_obj, all_opt,
                                    ind_var_in_plot=ind_var_in_plot,
                                    ind_var_between_plots=ind_var_between_plots,
                                    data_opt=data_opt,
                                    error_metric='correlation',
                                    global_dat_obj=data_obj,
                                    ind_var_dat_or_model='model',
                                    cmap=['r'],
                                    diagnostic_plots=diagnostic_plots)

ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_idmdc_parameters'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt)
