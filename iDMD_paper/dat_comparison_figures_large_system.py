# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:25:15 2019

@author: charl
"""

import test_data_utils as tdu
from pydmd import DMD, idmd, fbdmd, optimized_dmd
import matplotlib.pyplot as plt
import paper_settings as ps
from dmd_data import DMDData
import numpy as np
import random

random.seed(1415)
np.random.seed(random.randint(1, 10000))
to_save = True

###############################################################################
# Settings for all models
n_A = 20

all_obj = [DMD, idmd.iDMD, DMD, fbdmd.FbDMD, optimized_dmd.optDMD]
all_opt = [{'svd_rank':n_A},
           {'svd_rank':n_A, 'truncation':40, 'lambda_factor':0.6,
            'series_type':'geometric'},
           {'svd_rank':n_A, 'tlsq_rank':n_A},
           {'svd_rank':n_A},
           {'svd_rank':n_A}]
#all_obj = [optimized_dmd.optDMD]
#all_opt = [{'svd_rank':n_A}]
ind_var_in_plot = {'num_snapshots': [256, 512, 1028]}
small_ind_var_in_plot = {'num_snapshots': [1028]}
ind_var_between_plots = {'sigma': [0.01, 0.1]}

data_opt = {'n_A':n_A, 'dim_increase':50}
#data_opt = {'n_A':n_A, 'dim_increase':10}
data_obj = DMDData(**data_opt)
data_opt['num_datasets'] = 20
#data_opt['num_datasets'] = 1

fig_opt = {'ncols_panels': 2.0, 'ncols_paper': 2}

###############################################################################
# Figure 1a: Data comparisons with a small system
fig = tdu.plot_reconstruction_error(all_obj, all_opt,
                                    ind_var_in_plot=ind_var_in_plot,
                                    ind_var_between_plots=ind_var_between_plots,
                                    data_opt=data_opt,
                                    error_metric='correlation',
                                    global_dat_obj=data_obj)
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmd_large_corr'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt)
    
###############################################################################
# Figure 0b: Eig comparisons with a small system
#fig = tdu.plot_eig_error(all_obj, all_opt,
#                         ind_var_in_plot=ind_var_in_plot,
#                         ind_var_between_plots=ind_var_between_plots,
#                         data_opt=data_opt,
#                         global_dat_obj=data_obj)

# Also plot the complex plane
fig = tdu.plot_eig_error(all_obj, all_opt,
                         ind_var_in_plot=small_ind_var_in_plot,
                         ind_var_between_plots=ind_var_between_plots,
                         data_opt=data_opt,
                         global_dat_obj=data_obj,
                         plot_mode='complex')
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmd_large_eig'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt)
