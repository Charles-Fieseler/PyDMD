# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:09:55 2019

@author: charl

Produces several figures for comparing eigenvalues with different DMD methods,
WITH CONTROL
"""


import test_data_utils as tdu
from pydmd import DMDc, idmdc
import paper_settings as ps
import matplotlib.pyplot as plt
from dmd_data import DMDData
from math import sin, cos, pi
import numpy as np
import random

random.seed(1415)
np.random.seed(random.randint(1, 10000))
to_save = True

###############################################################################
# Global settings

num_datasets = 100
all_obj = [DMDc, idmdc.iDMDc]
#theta = 2.0*pi/60.0
#all_opt = [{'svd_rank':-1},
#           {'svd_rank':-1, 'truncation':40, 
#            'lambda_factor':0.6*complex(cos(theta), sin(theta)) }]
all_opt = [{'svd_rank':-1},
           {'svd_rank':-1, 'truncation':40, 'lambda_factor':0.6}]
ind_var_in_plot = {'num_snapshots': [128, 256, 512]}
small_ind_var_in_plot = {'num_snapshots': [512]}
ind_var_between_plots = {'sigma': [0.01, 0.1, 0.5]}

fig_opt = {'ncols_panels': 3.0, 'ncols_paper': 2}

###############################################################################
# Figure 2a: Eig comparisons with a small system
data_opt = {'n_A':2, 'n_B':1, 'dim_increase':0}
data_obj = DMDData(**data_opt)
data_opt['num_datasets'] = num_datasets
fig = tdu.plot_eig_error(all_obj, all_opt,
                         ind_var_in_plot=ind_var_in_plot,
                         ind_var_between_plots=ind_var_between_plots,
                         data_opt=data_opt,
                         global_dat_obj=data_obj)
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_small_eig_error'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt)

# Also plot the complex plane
fig = tdu.plot_eig_error(all_obj, all_opt,
                         ind_var_in_plot=small_ind_var_in_plot,
                         ind_var_between_plots=ind_var_between_plots,
                         data_opt=data_opt,
                         global_dat_obj=data_obj,
                         plot_mode='complex')
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_small_eig'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt)

#raise ValueError

###############################################################################
# Figure 2b: Eig comparisons with a small system
#data_opt = {'n_A':2, 'n_B':1, 'dim_increase':20}
#data_obj = DMDData(**data_opt)
#data_opt['num_datasets'] = num_datasets
#fig = tdu.plot_eig_error(all_obj, all_opt,
#                         ind_var_in_plot=ind_var_in_plot,
#                         ind_var_between_plots=ind_var_between_plots,
#                         data_opt=data_opt,
#                         global_dat_obj=data_obj)
#ps.set_plot_settings(fig)
#if to_save:
#    fname = 'fig2b'
#    plt.savefig(ps.foldername + fname, bbox_inches='tight')
#    #ps.save_current_figure(fname)
#
## Also plot the complex plane
#fig = tdu.plot_eig_error(all_obj, all_opt,
#                         ind_var_in_plot=small_ind_var_in_plot,
#                         ind_var_between_plots=ind_var_between_plots,
#                         data_opt=data_opt,
#                         global_dat_obj=data_obj,
#                         plot_mode='complex')

###############################################################################
# Figure 2a: Eig comparisons with a large system (known truncation)
all_opt = [{'svd_rank':6},
           {'svd_rank':6, 'truncation':40, 'lambda_factor':0.6}]
data_opt = {'n_A':4, 'n_B':2, 'dim_increase':20}
data_obj = DMDData(**data_opt)
data_opt['num_datasets'] = num_datasets
fig = tdu.plot_eig_error(all_obj, all_opt,
                         ind_var_in_plot=ind_var_in_plot,
                         ind_var_between_plots=ind_var_between_plots,
                         data_opt=data_opt,
                         global_dat_obj=data_obj)
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_known_eig_error'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    ps.save_current_figure(fname, **fig_opt)

# Also plot the complex plane
fig = tdu.plot_eig_error(all_obj, all_opt,
                         ind_var_in_plot=small_ind_var_in_plot,
                         ind_var_between_plots=ind_var_between_plots,
                         data_opt=data_opt,
                         global_dat_obj=data_obj,
                         plot_mode='complex')
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_known_eig'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt)
