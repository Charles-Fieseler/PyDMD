# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:51:56 2019

@author: charl
Produces several figures for comparing data to reconstructions with different 
DMD methods, WITH CONTROL
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
to_save = False

###############################################################################
# Settings for all models
all_obj = [DMDc, idmdc.iDMDc]
all_opt = [{'svd_rank':-1},
           {'svd_rank':-1, 'truncation':40, 'lambda_factor':0.6}]
#all_opt = [{'svd_rank':0},
#           {'svd_rank':0, 'truncation':40, 'lambda_factor':0.6}]
#all_opt = [{'svd_rank':-1, 'tlsq_rank':2},
#           {'svd_rank':-1, 'tlsq_rank':2, 'truncation':40, 'lambda_factor':0.6}]
ind_var_in_plot = {'num_snapshots': [256, 512, 1028]}
ind_var_between_plots = {'sigma': [0.01, 0.1, 0.5]}

# Settings for example trajectory plots
tspan = [300, 500]

fig_opt1 = {'ncols_panels': 3.0, 'ncols_paper': 2}
fig_opt2 = {'ncols_panels': 1.0, 'ncols_paper': 2}

###############################################################################
# Figure 1a: Data comparisons with a small system
#data_opt = {'n_A':2, 'n_B':1, 'num_datasets':20, 'dim_increase':0}
data_opt = {'n_A':2, 'n_B':1, 'dim_increase':0}
data_obj = DMDData(**data_opt)
data_opt['num_datasets'] = 20
fig = tdu.plot_reconstruction_error(all_obj, all_opt,
                                    ind_var_in_plot=ind_var_in_plot,
                                    ind_var_between_plots=ind_var_between_plots,
                                    data_opt=data_opt,
                                    error_metric='correlation',
                                    global_dat_obj=data_obj)
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_small_corr'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    ps.save_current_figure(fname, **fig_opt1)

# Also produce an example iDMDc and DMDc fit
ex_noisy, ex_true = data_obj.produce_data(
        sigma=ind_var_between_plots['sigma'][1])

ex_idmdc = idmdc.iDMDc(**all_opt[1])
ex_idmdc.fit(ex_noisy, data_obj.U[:, :-1])
while True:
    ex_dmdc = DMDc(**all_opt[0])
    ex_dmdc.fit(ex_noisy, data_obj.U[:, :-1])
    print(ex_dmdc.is_stable)
    if ex_dmdc.is_stable:
        break
    else:
        ex_noisy, ex_true = data_obj.produce_data(
                sigma=ind_var_between_plots['sigma'][1])

fig2 = plt.figure()
tpu.plot_multiple_models([ex_dmdc, ex_idmdc], ex_noisy)
plt.xlim(tspan)
ps.set_plot_settings(fig2)
if to_save:
    fname = 'fig_dmdc_small_ex'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt2)

###############################################################################
# Figure 1b: Data comparison in a medium system
#all_opt = [{'svd_rank':3},
#           {'svd_rank':3, 'truncation':40, 'lambda_factor':0.6}]
##all_opt = [{'svd_rank':-1, 'tlsq_rank':3},
##           {'svd_rank':-1, 'tlsq_rank':3, 'truncation':40, 'lambda_factor':0.6}]
#data_opt = {'n_A':2, 'n_B':1, 'dim_increase':20}
#data_obj = DMDData(**data_opt)
#data_opt['num_datasets'] = 20
#fig = tdu.plot_reconstruction_error(all_obj, all_opt,
#                                    ind_var_in_plot=ind_var_in_plot,
#                                    ind_var_between_plots=ind_var_between_plots,
#                                    data_opt=data_opt,
#                                    error_metric='correlation',
#                                    global_dat_obj=data_obj)
#ps.set_plot_settings(fig)
#if to_save:
#    fname = 'fig1b'
#    plt.savefig(ps.foldername + fname, bbox_inches='tight')
#    ps.save_current_figure(fname)
#
## Also produce an example iDMDc and DMDc fit
#ex_noisy, ex_true = data_obj.produce_data(
#        sigma=ind_var_between_plots['sigma'][1])
#
#ex_idmdc = idmdc.iDMDc(**all_opt[1])
#ex_idmdc.fit(ex_noisy, data_obj.U[:, :-1])
#while True:
#    ex_dmdc = DMDc(**all_opt[0])
#    ex_dmdc.fit(ex_noisy, data_obj.U[:, :-1])
#    print(ex_dmdc.is_stable)
#    if ex_dmdc.is_stable:
#        break
#    else:
#        ex_noisy, ex_true = data_obj.produce_data(
#                sigma=ind_var_between_plots['sigma'][1])
#
#fig2 = plt.figure()
#tpu.plot_multiple_models([ex_idmdc, ex_dmdc], ex_noisy)
#ps.set_plot_settings(fig2)
#if to_save:
#    fname = 'fig1b2'
#    plt.savefig(ps.foldername + fname, bbox_inches='tight')
#    ps.save_current_figure(fname)

###############################################################################
# Figure 1c: Data comparison in a large system, with CORRECT truncation
#all_opt = [{'svd_rank':6},
#           {'svd_rank':6, 'truncation':40, 'lambda_factor':0.6}]
all_opt = [{'svd_rank':25},
           {'svd_rank':25, 'truncation':40, 'lambda_factor':0.6}]
#data_opt = {'n_A':4, 'n_B':2, 'dim_increase':20}
data_opt = {'n_A':20, 'n_B':5, 'dim_increase':50}
data_obj = DMDData(**data_opt)
data_opt['num_datasets'] = 20
fig = tdu.plot_reconstruction_error(all_obj, all_opt,
                                    ind_var_in_plot=ind_var_in_plot,
                                    ind_var_between_plots=ind_var_between_plots,
                                    data_opt=data_opt,
                                    error_metric='correlation',
                                    global_dat_obj=data_obj)
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_known_corr'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    ps.save_current_figure(fname, **fig_opt1)

# Also produce an example iDMDc and DMDc fit
ex_noisy, ex_true = data_obj.produce_data(
        sigma=ind_var_between_plots['sigma'][1])

ex_idmdc = idmdc.iDMDc(**all_opt[1])
ex_idmdc.fit(ex_noisy, data_obj.U[:, :-1])
while True:
    ex_dmdc = DMDc(**all_opt[0])
    ex_dmdc.fit(ex_noisy, data_obj.U[:, :-1])
    print(ex_dmdc.is_stable)
    if ex_dmdc.is_stable:
        break
    else:
        ex_noisy, ex_true = data_obj.produce_data(
                sigma=ind_var_between_plots['sigma'][1])

fig2 = plt.figure()
tpu.plot_multiple_models([ex_dmdc, ex_idmdc], ex_noisy)
plt.xlim(tspan)

ps.set_plot_settings(fig2)
if to_save:
    fname = 'fig_dmdc_known_ex'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt2)

###############################################################################
# Figure 1c: Data comparison in a large system, with GAVISH-DONOHO truncation
all_opt = [{'svd_rank':0},
           {'svd_rank':0, 'truncation':40, 'lambda_factor':0.6}]
fig = tdu.plot_reconstruction_error(all_obj, all_opt,
                                    ind_var_in_plot=ind_var_in_plot,
                                    ind_var_between_plots=ind_var_between_plots,
                                    data_opt=data_opt,
                                    error_metric='correlation',
                                    global_dat_obj=data_obj)
ps.set_plot_settings(fig)
if to_save:
    fname = 'fig_dmdc_unknown_corr'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    ps.save_current_figure(fname, **fig_opt1)

# Also produce an example iDMDc and DMDc fit
ex_noisy, ex_true = data_obj.produce_data(
        sigma=ind_var_between_plots['sigma'][1])

ex_idmdc = idmdc.iDMDc(**all_opt[1])
ex_idmdc.fit(ex_noisy, data_obj.U[:, :-1])
while True:
    ex_dmdc = DMDc(**all_opt[0])
    ex_dmdc.fit(ex_noisy, data_obj.U[:, :-1])
    print(ex_dmdc.is_stable)
    if ex_dmdc.is_stable:
        break
    else:
        ex_noisy, ex_true = data_obj.produce_data(
                sigma=ind_var_between_plots['sigma'][1])

fig2 = plt.figure()
tpu.plot_multiple_models([ex_dmdc, ex_idmdc], ex_noisy)
plt.xlim(tspan)

ps.set_plot_settings(fig2)
if to_save:
    fname = 'fig_dmdc_unknown_ex'
    plt.savefig(ps.foldername + fname, bbox_inches='tight')
    plt.savefig(ps.foldername + fname, bbox_inches='tight', format='eps')
    ps.save_current_figure(fname, **fig_opt2)
