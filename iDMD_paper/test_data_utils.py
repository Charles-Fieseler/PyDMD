# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:47:58 2019

@author: charl
"""

#from math import sqrt
#from scipy.stats import ortho_group
from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
import random
from dmd_data import DMDData
from pydmd import optdmd # Needs special clean-up
import test_plot_utils as tpu
import scipy.io as scio


###############################################################################
# Creating dynamic matrices
def create_dynamics(n, arg_bounds=[1.0, 1.0], phase_bounds=[0.0, pi/4],
                    real_matrix=True):
    """
    Creates a random linear dynamic matrix (A) with bounded eigenvalues
    """
    m = np.random.rand(n, n)
    vecs = np.linalg.eig(m-m.T)[1]  # Complex eigenvalues
    
    if real_matrix:
        assert n%2==0
        n = int(n/2)
        
    eigs = []
    for i in range(n):
#        x = random.uniform(0.5, 1.0)  # Don't want to have too fast an omega
        theta = random.uniform(phase_bounds[0], phase_bounds[1])
        sign = 2*random.randint(0,1)-1
        r = random.uniform(arg_bounds[0], arg_bounds[1])
#        this_eig = complex(x, sign*sqrt(r**2-x**2))
        this_eig = complex(r*cos(theta), sign*r*sin(theta))
        eigs.append(this_eig)
        if real_matrix:
            eigs.append(np.conj(this_eig))
    
    A = vecs @ np.diag(eigs) @ np.linalg.inv(vecs)
    if real_matrix:
        A = A.real
#    A = vecs @ np.diag(eigs) @ np.conj(vecs.T)
    return A, eigs


def create_control_system(sigma=0.0, n_A=2, n_B=1, m=200, U_sparsity=0.99,
                          dim_increase=0):
    A, true_eigs = create_dynamics(n_A)
    B = np.random.rand(n_A, n_B)
    U = create_control_signal(n_B, m-1, U_sparsity)

    X = create_noisy_data_control(sigma, n_A, n_B, A=A, B=B, U=U, m=m,
                                  dim_increase=dim_increase)[0]

    return A, B, U, X


###############################################################################
# Creating Noisy Data
def create_noisy_data_simple(sigma=0.0, calc_eigs=False):
    mu = 0.
    m = 200  # number of snapshot
    n = 2
    noise = np.random.normal(mu, sigma, m)  # gaussian noise
    A = np.array([[1., 1.], [-1., 2.]])
    A /= np.sqrt(3)
    if calc_eigs:
        true_eigs, true_vecs = np.linalg.eig(A)
    else:
        true_eigs = None
    X = np.zeros((n, m))
    X[:, 0] = np.array([0.5, 1.])
    # evolve the system and perturb the data with noise
    for k in range(1, m):
        X[:, k] = A.dot(X[:, k - 1])
        X[:, k - 1] += noise[k - 1]
    return X, true_eigs


def create_noisy_data(sigma=0.0, n_A=2, A=None, 
                      m=200,
                      dim_increase=0,
                      calc_eigs=False):
#        sigma=0.0, n=2, calc_eigs=False, A=None, m=200):
    noise = np.random.normal(0.0, sigma, m)  # gaussian noise
    if A is None:
        A, true_eigs = create_dynamics(n_A)
    else:
        n = np.shape(A)[0]
    if calc_eigs:
        true_eigs, true_vecs = np.linalg.eig(A)
    else:
        true_eigs = None
    X = np.zeros((n, m))
    X_noise = np.zeros_like(X)
#    X[:, 0] = complex(np.random.rand(n), np.random.rand(n))
    X[:, 0] = np.random.rand(n)
    # evolve the system and perturb the data with noise
    for k in range(1, m):
#        X[:, k] = A.dot(X[:, k - 1])
        X[:, k] = A @ X[:, k - 1]
        X_noise[:, k - 1] = X[:, k - 1] + noise[k - 1]
    if dim_increase > 0:
        C = np.random.rand(n_A+dim_increase, n_A)
        X_noise = C @ X_noise
        X = C @ X
    return X_noise, X, true_eigs


###############################################################################
# Control Systems (DMDc)
def create_control_signal(n, m, sparsity):
    U = np.random.rand(n, m)
    U[U<sparsity] = 0
    return U


def create_noisy_data_control(sigma=0.0, n_A=2, n_B=2, A=None, B=None, 
                              m=200,
                              dim_increase=0,
                              U=None,
                              U_sparsity=0.9,
                              calc_eigs=False):
    """
    Creates a noisy dataset produced via linear dynamics with a sparse control
    signal, for use with DMDc
    """
    noise = np.random.normal(0.0, sigma, m)  # gaussian noise
    if A is None:
        A, true_eigs = create_dynamics(n_A)
    else:
        n_A = np.shape(A)[0]
    if B is None:
        B = np.random.rand(n_A, n_B)
    if U is None:
        U = create_control_signal(n_B, m, U_sparsity)

    if calc_eigs:
        true_eigs, true_vecs = np.linalg.eig(A)
    else:
        true_eigs = None
    X = np.zeros((n_A, m))
    X_noise = np.zeros((n_A, m))
    X[:, 0] = np.random.rand(n_A)
    X_noise[:, 0] = X[:, 0] + noise[0]
    # evolve the system and perturb the data with noise
    for k in range(1, m):
        X[:, k] = A @ X[:, k-1] + B @ U[:, k-1]
        X_noise[:, k] = X[:, k] + noise[k]
    if dim_increase > 0:
        C = np.random.rand(n_A+dim_increase, n_A)
        X_noise = C @ X_noise
        X = C @ X
    return X_noise, X, true_eigs


###############################################################################
# Creating groups of models
def calc_noisy_datasets(num_pts=10, sigma=0.1, n_A=2, min_eig=1.0, m=200):
    # Creates a number of models with different datasets of noise level "sigma"
    all_dat = []
    all_true_dat = []
    true_A, true_eigs = create_dynamics(n_A, arg_bounds=[min_eig, 1.0])
    for i in range(num_pts):
        noisy_dat, true_dat, tmp = create_noisy_data(
                sigma, calc_eigs=False, A=true_A, m=m)
        all_dat.append(noisy_dat)
        all_true_dat.append(true_dat)

    return all_dat, all_true_dat, np.array(true_eigs)


def calc_noisy_datasets_control(num_pts=10, sigma=0.1, n_A=2, n_B=1,
                                min_eig=1.0, m=200, U_sparsity=0.99,
                                dim_increase=0):
    # Creates a number of different datasets of noise level "sigma"
    all_dat = []
    all_true_dat = []
    true_A, true_B, U, X = create_control_system(sigma=sigma,
                                                 n_A=n_A, n_B=n_B,
                                                 m=m, U_sparsity=U_sparsity)
    true_eigs = np.linalg.eig(true_A)[0]
    for i in range(num_pts):
        noisy_dat, true_dat, tmp = create_noisy_data_control(
                sigma, calc_eigs=False,
                A=true_A, B=true_B, U=U,
                m=m, dim_increase=dim_increase)
        all_dat.append(noisy_dat)
        all_true_dat.append(true_dat)

    return all_dat, all_true_dat, np.array(true_eigs), U


def calc_eig_cloud(all_dat, model_obj, model_opt, 
                   U=None, data_mode='eigs',
                   diagnostic_plots=False):
    # Loops through datasets and fits a model object to each dataset
    #   model_opt is a dictionary of options to pass to each model
    # Returns the eigenvalues OR reconstructions, based on data_mode

    all_eigs = []
    label = None
    this_model = model_obj(**model_opt)
    for dat in all_dat:
#        this_model = model_obj(**model_opt)
        if U is None:
            this_model.fit(dat)
        else:
            this_model.fit(dat, U)
        if diagnostic_plots:
#            plt.figure()
#            plt.plot(dat[0], label='dat')
#            plt.title('Dat')
            plt.figure()
            tpu.plot_multiple_models([this_model], truth=dat,
                                     plot_mode=data_mode)
            diagnostic_plots = False  # Only plot one
#            plt.title(this_model.label_for_plots)
        if label is None:
            label = this_model.label_for_plots
        if data_mode is 'eigs':
            all_eigs.append(this_model._eigs)
        elif data_mode is 'reconstructed_data':
            if U is None:
                all_eigs.append(this_model.reconstructed_data)
            else:
                all_eigs.append(this_model.reconstructed_data(U))
        else:
            raise ValueError("Unrecognized data mode")
    
    if model_obj is optdmd.optDMD:
        this_model.clean_up()

    return np.array(all_eigs), label


def calc_eig_error(dmd_eig, true_eig, error_metric=None):
    # Makes sure that the eigenvalues being compared are the closest to each
    # other, i.e. not out of order
    #   
    if error_metric is not 'L2':
        raise ValueError("Other metrics not yet implemented")

#    my_norm = abs
    all_err = []
    all_ind = []
    max_error = 1.0  # If the error is larger, it wasn't found at all
    for t in true_eig:
        min_error = max_error
        min_index = None
        for i in range(len(dmd_eig)):
            e = dmd_eig[i]
#            err = my_norm(e, t)
            err = abs(e-t)
            if err < min_error:
                min_error = err
                min_index = i
        all_err.append(min_error)
        all_ind.append(min_index)
    
    repeats = [x for x in all_ind if 
               (all_ind.count(x) > 1 and x is not None)]
    if len(repeats)>0:
#        raise ValueError("Duplicated learned eigenvalues; TODO")
        print("Duplicated learned eigenvalues; TODO")
    
    return sum(all_err), all_ind


def calc_reconstruction_error(recon_dat, true_dat,
                              error_metric='L2', max_error = 1e2,
                              ind=None):
    # Calculates the error between two time series. Default uses L2 norm

    if ind is not None:
        recon_dat = recon_dat[:, ind]
        true_dat = true_dat[:, ind]

    if error_metric is 'L2':
        err = abs(pow(recon_dat - true_dat, 2.0))
        err = np.mean(err) / np.sum(pow(true_dat,2.0))
    elif error_metric is 'correlation':
        err = []
        for r, t in zip(recon_dat, true_dat):
            err.append(np.corrcoef(r, t))
        err = np.mean(err)
    else:
        raise ValueError("Other metrics not yet implemented")
#    err = np.mean(err, 0)  # Across input channels
#    err = np.mean(err)  # Across time points and channels

    if err > max_error:
        err = np.inf
    return err


###############################################################################
# Plotting
def plot_eig_clouds(all_model_objs, all_model_opts, sigma=0.1, n=2):
    # Loops through a list of model objects with options, creates eigenvalue
    # clouds, and plots them

    all_dat, true_dat, true_eigs = calc_noisy_datasets(sigma=sigma, n=n)
    all_labels = []
    all_eigs = []
    first_plot = True
    for f, opt in zip(all_model_objs, all_model_opts):
        these_eigs, label = calc_eig_cloud(all_dat, f, opt)
        all_eigs.append(these_eigs)
        all_labels.append(label)
        if first_plot:
            model = f(**opt)
            model.fit(all_dat[0])
            model.plot_eigs()
            plt.plot(true_eigs.real, true_eigs.imag, '+k', ms=20.0)
            all_labels.append('True')
            first_plot = False

    plot_opt = ['ob', 'or', 'om', 'og', 'ok']
    for i_dat in range(len(all_dat)):
        for i_model in range(len(all_eigs)):
            if i_dat==0 and i_model==0:
                continue
            e = all_eigs[i_model][i_dat]
            plt.plot(e.real, e.imag, plot_opt[i_model])

    plt.legend(all_labels)
    plt.title('Noise level is %.1f' % sigma)

    return


def plot_eig_error(all_model_objs, all_model_opts,
                   ind_var_in_plot={'m': [128, 256, 512]},
                   ind_var_between_plots={'sigma': [0.01, 0.2]},
                   data_opt={'n_A': 2},
                   global_dat_obj=None,
                   plot_mode='boxplot'):
    """
    Plots the L2 error of the eigenvalues for different models, corresponding
    options, and a dict of independent variables
    
    :param all_model_objs: a list of DMD or DMDc class objects to be used to
        analyze the data
    :param all_model_opts: options for the above models, also a list
    :param ind_var_in_plot: a dict; key is a variable as passed to the 
        dataset creation function, 'calc_noisy_datasets'; the value is the
        range of values created. This will be the x-axis within the subplots
    :param ind_var_between_plots: a dict, same as above; key is the independent
        variable between adjacent subplots
    :param data_opt: other options to be passed for the data production; can't
        overlap with the independent variables above
        
    See also: plot_reconstruction_error
    """

    # Sanity checks and initial settings
    num_models = len(all_model_objs)
    if len(all_model_opts) != num_models:
        raise ValueError("Number of options and number of models do not match")
    if 'n_B' in data_opt:
        use_control = True
    else:
        use_control = False
    if not (data_opt.keys().isdisjoint(ind_var_in_plot.keys())):
        raise ValueError('Cannot fix data options that will be iterated over')
    if not (data_opt.keys().isdisjoint(ind_var_between_plots.keys())):
        raise ValueError('Cannot fix data options that will be iterated over')

    # First calculate the datasets
    f = calc_subplot_datasets
    all_dat, tmp, true_eigs, all_U = f(data_opt,
                                       use_control,
                                       ind_var_between_plots,
                                       ind_var_in_plot,
                                       global_dat_obj=global_dat_obj)
    # Produce models and get eigenvalues from them
    all_labels, all_eigs = calc_subplot_models(all_model_objs,
                                               all_model_opts,
                                               all_dat,
                                               all_U,
                                               ind_var_in_plot,
                                               data_mode='eigs')
    # Last, actually plot using subplots
    my_cmap = ['b', 'r', 'g', 'm', 'k']
    if plot_mode is 'scatter':
        f = plot_subplots_scatter
    elif plot_mode is 'boxplot':
        f = plot_subplots_boxplot
    elif plot_mode is 'complex':
        f = plot_subplots_complex
    fig = f(my_cmap,
            all_model_objs,
            true_eigs, all_eigs, all_labels,
            ind_var_between_plots, ind_var_in_plot)[0]

    return fig


def plot_reconstruction_error(all_model_objs, all_model_opts,
                              ind_var_in_plot={'m': [128, 256, 512]},
                              ind_var_dat_or_model='dat',
                              ind_var_between_plots={'sigma': [0.01, 0.2]},
                              data_opt={'n_A': 2},
                              error_metric='L2',
                              diagnostic_plots=False,
                              plot_mode='boxplot',
                              global_dat_obj=None,
                              cmap = ['b', 'r', 'g', 'm', 'k']):
    """
    Plots the L2 error of the reconstructions for different models, 
    corresponding options, and a dict of independent variables.
    
    :param all_model_objs: a list of DMD or DMDc class objects to be used to
        analyze the data
    :param all_model_opts: options for the above models, also a list
    :param ind_var_in_plot: a dict; key is a variable as passed to the
        dataset creation function, 'calc_noisy_datasets'; the value is the
        range of values created. This will be the x-axis within the subplots
    :param ind_var_dat_or_model: string; either 'dat' for the independent
        variable changing the dataset, or 'model' for change in the models
    :param ind_var_between_plots: a dict, same as above; key is the independent
        variable between adjacent subplots
    :param data_opt: other options to be passed for the data production; can't
        overlap with the independent variables above

    See also: plot_eig_error
    """

    # Sanity checks and initial settings
    num_models = len(all_model_objs)
    if len(all_model_opts) != num_models:
        raise ValueError("Number of options and number of models do not match")
    if 'n_B' in data_opt:
        use_control = True
    else:
        use_control = False
    if not (data_opt.keys().isdisjoint(ind_var_in_plot.keys())):
        raise ValueError('Cannot fix data options that will be iterated over')
    if not (data_opt.keys().isdisjoint(ind_var_between_plots.keys())):
        raise ValueError('Cannot fix data options that will be iterated over')

    # First calculate the datasets
    #   Note: 'all_dat' has noise and 'true_dat' doesn't
    f = calc_subplot_datasets
    all_dat, true_dat, true_eigs, all_U = f(data_opt,
                                            use_control,
                                            ind_var_between_plots,
                                            ind_var_in_plot,
                                            global_dat_obj=global_dat_obj,
                                    ind_var_dat_or_model=ind_var_dat_or_model)
    # Produce models and get eigenvalues from them
    f = calc_subplot_models
    all_labels, all_recon = f(all_model_objs,
                              all_model_opts,
                              all_dat,
                              all_U,
                              ind_var_in_plot,
                              data_mode='reconstructed_data',
                              diagnostic_plots=diagnostic_plots,
                              ind_var_dat_or_model=ind_var_dat_or_model)
    # Equalize the length of the reconstructions
    if 'num_snapshots' in ind_var_in_plot:
        ind = list(range(min(ind_var_in_plot['num_snapshots'])))
    else:
        ind = None

    # Last, actually plot using subplots
    if plot_mode is 'scatter':
        f = plot_subplots_scatter
    elif plot_mode is 'boxplot':
        f = plot_subplots_boxplot
    fig = f(cmap, all_model_objs,
            true_dat, all_recon, all_labels,
            ind_var_between_plots, ind_var_in_plot,
            plot_mode='reconstructed_data',
            error_metric=error_metric,
            ind=ind)[0]

    return fig


def plot_improvement_hist(all_model_objs, all_model_opts,
                          ind_var_in_plot={'m': [256]},
                          ind_var_dat_or_model='dat',
                          ind_var_between_plots={'sigma': [0.1]},
                          data_opt={'n_A': 2},
                          error_metric='L2',
                          dat_obj_list=None,
                          cmap = ['b', 'r', 'g', 'm', 'k']):
    """
    Plots a histogram of the error of the reconstructions for models
    corresponding to different underlying systems.
    
    :param all_model_objs: a list of DMD or DMDc class objects to be used to
        analyze the data
    :param all_model_opts: options for the above models, also a list
    :param ind_var_in_plot: a dict; key is a variable as passed to the
        dataset creation function, 'calc_noisy_datasets'; should be a single
        value, not a list
    :param ind_var_dat_or_model: string; either 'dat' for the independent
        variable changing the dataset, or 'model' for change in the models
    :param ind_var_between_plots: a dict, same as above; key is the independent
        variable between adjacent subplots; TODO: allow a list
    :param data_opt: other options to be passed for the data production; can't
        overlap with the independent variables above

    See also: plot_eig_error
    """

    # Sanity checks and initial settings
    num_models = len(all_model_objs)
    if len(all_model_opts) != num_models:
        raise ValueError("Number of options and number of models do not match")
    if 'n_B' in data_opt:
        use_control = True
    else:
        use_control = False
    if not (data_opt.keys().isdisjoint(ind_var_in_plot.keys())):
        raise ValueError('Cannot fix data options that will be iterated over')
    if not (data_opt.keys().isdisjoint(ind_var_between_plots.keys())):
        raise ValueError('Cannot fix data options that will be iterated over')

    if dat_obj_list is None:
        raise ValueError("Must pass a list of data objects")
    num_systems = len(dat_obj_list)
    
    # First calculate the datasets
    #   Note: 'all_dat' has noise and 'true_dat' doesn't
    f = calc_subplot_datasets
    all_dat_list, true_dat_list, true_eigs_list, all_U_list = [], [], [], []
    for i in range(num_systems):
        all_dat, true_dat, true_eigs, all_U = f(data_opt,
                                                use_control,
                                                ind_var_between_plots,
                                                ind_var_in_plot,
                                                global_dat_obj=dat_obj_list[i],
                                    ind_var_dat_or_model=ind_var_dat_or_model)
        all_dat_list.append(all_dat)
        # TODO: implement these
        true_dat_list.append(true_dat)
#        true_eigs_list.append(true_eigs)
        all_U_list.append(all_U)
        
    # Produce models and get eigenvalues from them
    f = calc_subplot_models
    all_labels_list, all_recon_list = [], []
    for i in range(num_systems):
        all_labels, all_recon = f(all_model_objs,
                                  all_model_opts,
                                  all_dat_list[i],
                                  all_U_list[i],
                                  ind_var_in_plot,
                                  data_mode='reconstructed_data',
                                  ind_var_dat_or_model=ind_var_dat_or_model)
        all_labels_list.append(all_labels)
        all_recon_list.append(all_recon)

    # Last, plot histograms
    f = plot_subplots_histogram
    fig = f(cmap, all_model_objs,
            true_dat_list, all_recon_list, all_labels_list,
            ind_var_between_plots, ind_var_in_plot,
            plot_mode='reconstructed_data',
            error_metric=error_metric)[0]

    return fig


###############################################################################
# Helper functions for main plotting functions

def calc_subplot_datasets(data_opt, use_control,
                          ind_var_between_plots, ind_var_in_plot,
                          global_dat_obj=None,
                          ind_var_dat_or_model=None,
                          data_filename='tmp'):
    # Given the independent variables within and between subplots, calculate
    # clouds of eigenvalues for each x value in each subplot
    
    all_dat, all_dat_true, true_eigs, all_U = [], [], [], []
    btw_plot_key = list(ind_var_between_plots.keys())[0]
    for btw_plot_val in ind_var_between_plots[btw_plot_key]:
        in_plot_key = list(ind_var_in_plot.keys())[0]
        subplot_dat, subplot_true, subplot_eigs, subplot_U = [], [], [], []
        if global_dat_obj is None:
            subplot_opt = data_opt.copy()
            subplot_opt.update({btw_plot_key: btw_plot_val})
            # Use new object to produce the data for the entire subplot
            dat_obj = DMDData(**subplot_opt)
        else:
            dat_obj = global_dat_obj
            setattr(dat_obj, btw_plot_key, btw_plot_val)
        for in_plot_val in ind_var_in_plot[in_plot_key]:
            if ind_var_dat_or_model is 'dat':
                ind_var_opt = {in_plot_key: in_plot_val}
                tmp_dat, tmp_true = dat_obj.produce_data_cloud(**ind_var_opt)
            else:
                tmp_dat, tmp_true = dat_obj.produce_data_cloud()
            if use_control:
                sz = np.shape(tmp_dat)
                subplot_U.append(dat_obj.U[:, range(sz[2]-1)])
            else:
                subplot_U.append(None)
            subplot_dat.append(tmp_dat)
            subplot_true.append(tmp_true)
            subplot_eigs.append(dat_obj.eigs)
        all_dat.append(subplot_dat)
        all_dat_true.append(subplot_true)
        true_eigs.append(subplot_eigs)
        all_U.append(subplot_U)
    
#    if data_filename is not None:
#        scio.savemat(data_filename,
#                     {'X00':all_dat[0][0],
#                      'X01':all_dat[0][1],
#                      'X02':all_dat[0][2],
#                      'X10':all_dat[1][0],
#                      'X11':all_dat[1][1],
#                      'X12':all_dat[1][2]})

    return all_dat, all_dat_true, true_eigs, all_U


def calc_subplot_models(all_model_objs, all_model_opts, all_dat, all_U,
                        ind_var_in_plot,
                        data_mode='eigs', diagnostic_plots=False,
                        ind_var_dat_or_model=None):
    # Calculates models, given datasets for multiple subplots. Returns the
    # eigenvalue clouds OR reconstructions and the associated model labels

    all_labels, all_out = [], []
    for f, opt in zip(all_model_objs, all_model_opts):
        subplot_out, subplot_labels = [], []
        for subplot_dat, subplot_U in zip(all_dat, all_U):
            xval_out, xval_labels = [], []
            for i_x in range(len(subplot_dat)):
                xval_dat = subplot_dat[i_x]
                # Note: xval_U is 'None' if using regular DMD
                xval_U = subplot_U[i_x]
                this_opt = opt
                if ind_var_dat_or_model is 'model':
                    in_plot_key = list(ind_var_in_plot.keys())[0]
                    this_opt.update({
                            in_plot_key: ind_var_in_plot[in_plot_key][i_x]})
                these_out, label = calc_eig_cloud(xval_dat, f, this_opt,
                                                  U=xval_U,
                                                  data_mode=data_mode,
                                          diagnostic_plots=diagnostic_plots)
                if diagnostic_plots and ind_var_dat_or_model is 'model':
                    # Otherwise there is no title
                    plt.title('%s=%.2f' % 
                              (in_plot_key, this_opt[in_plot_key]))
                xval_out.append(these_out)
                xval_labels.append(label)
            subplot_out.append(xval_out)
            subplot_labels.append(xval_labels)
        all_out.append(subplot_out)
        all_labels.append(subplot_labels)

    return all_labels, all_out


def plot_subplots_scatter(my_cmap, all_model_objs,
                          true_plot_dat, all_plot_dat, all_labels,
                          ind_var_between_plots, ind_var_in_plot,
                          plot_mode='eigs',
                          error_metric='L2',
                          latex_title=True,
                          ind=None):
    # Plots subplots with different columns

    subplot_key = list(ind_var_between_plots.keys())[0]
    subplot_vec = ind_var_between_plots[subplot_key]
    num_subplots = len(subplot_vec)
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots)
#    plt.ylabel('Error')

    k = list(ind_var_in_plot.keys())[0]
    num_columns = len(ind_var_in_plot[k])
#    plt.xlabel(k)
    x_vec = ind_var_in_plot[k]

    x_jitter = 0.2*np.random.rand(len(all_model_objs))*np.mean(np.diff(x_vec))
    for i_subplot in range(num_subplots):
        ax = axes[i_subplot]
        for i_model in range(len(all_model_objs)):
            for i_x in range(num_columns):
                # Get eigs or reconstruction for each data instance
                e = all_plot_dat[i_model][i_subplot][i_x]
                if i_x == 0:
                    this_label = all_labels[i_model][i_subplot][0].replace(
                            '_', ';')
                else:
                    this_label = ''
                t = true_plot_dat[i_subplot][i_x]
                # Calculate error (y values)
                y = []
                for e_one_model in e:
                    if plot_mode is 'eigs':
                        y_tmp = calc_eig_error(e_one_model, t, error_metric)[0]
                    elif plot_mode is 'reconstructed_data':
                        y_tmp = calc_reconstruction_error(e_one_model,
                                                          t, error_metric,
                                                          ind=ind)
                    else:
                        raise ValueError("Unknown plot mode")
                    y.append(y_tmp)
                # Finally plot
                x = x_vec[i_x] * np.ones_like(y) + x_jitter[i_model]
                ax.plot(x, y, my_cmap[i_model]+'o', alpha=0.2)
                ax.plot(np.mean(x), np.mean(np.mean(y)), my_cmap[i_model] + '+',
                        label=this_label, ms=20.0)
                if error_metric is 'correlation':
                    ax.set_ylim(0.4, 1)
        if latex_title:
            ax.set_title("$\%s=%.2f$" % (subplot_key, subplot_vec[i_subplot]))
        else:
            ax.set_title("%s=%.2f" % (subplot_key, subplot_vec[i_subplot]))
        ax.legend()
        ax.set_xlabel(k)
        ax.set_ylabel(error_metric)
    
    return fig, axes


def plot_subplots_boxplot(my_cmap, all_model_objs,
                          true_plot_dat, all_plot_dat, all_labels,
                          ind_var_between_plots, ind_var_in_plot,
                          plot_mode='eigs',
                          error_metric='L2',
                          latex_title=True,
                          ind=None):
    # Plots subplots with different boxplots

    subplot_key = list(ind_var_between_plots.keys())[0]
    subplot_vec = ind_var_between_plots[subplot_key]
    num_subplots = len(subplot_vec)
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots)
#    plt.ylabel('Error')

    k = list(ind_var_in_plot.keys())[0]
    num_columns = len(ind_var_in_plot[k])
    x_vec = ind_var_in_plot[k]

    for i_subplot in range(num_subplots):
        ax = axes[i_subplot]
        # Calculate the data, then plot outside the loops
        #######################################################################
        tmp1, tmp2, tmp3 = calc_boxplot_data(all_model_objs,
                                               all_labels,
                                               all_plot_dat,
                                               true_plot_dat,
                                               error_metric,
                                               num_columns,
                                               plot_mode,
                                               ind,
                                               x_vec,
                                               i_subplot)
        box_dat, tick_labels, group_labels = tmp1, tmp2, tmp3
#        box_dat.append(tmp1)
#        group_labels.append(tmp2)
#        tick_labels.append(tmp3)
        #######################################################################
#        box_dat, group_labels, tick_labels = [], [], []
#        for i_model in range(len(all_model_objs)):
#            model_dat = []
#            for i_x in range(num_columns):
#                # Get eigs or reconstruction for each data instance
#                e = all_plot_dat[i_model][i_subplot][i_x]
#                if i_x == 0:
#                    group_labels.append(
#                            all_labels[i_model][i_subplot][0].replace(
#                                    '_', ';'))
#                t = true_plot_dat[i_subplot][i_x]
#                y = []
#                for e_one_model in e:
#                    if plot_mode is 'eigs':
#                        y_tmp = calc_eig_error(e_one_model, t, error_metric)[0]
#                    elif plot_mode is 'reconstructed_data':
#                        y_tmp = calc_reconstruction_error(e_one_model,
#                                                          t, error_metric,
#                                                          ind=ind)
#                    else:
#                        raise ValueError("Unknown plot mode")
#                    y.append(y_tmp)
#                model_dat.append(np.array(y)[np.isfinite(y)])
#                if i_model == 0:
#                    tick_labels.append('%d' % x_vec[i_x])
#            box_dat.append(model_dat)
        #######################################################################
        
        # Finally plot
        plt.sca(ax)
        tpu.plot_grouped_boxplots(box_dat, group_labels, tick_labels,
                                  cmap=my_cmap, show_legend=(i_subplot==0))
        if latex_title:
            ax.set_title("$\%s=%.2f$" % (subplot_key, subplot_vec[i_subplot]))
        else:
            ax.set_title("%s=%.2f" % (subplot_key, subplot_vec[i_subplot]))
        if i_subplot==0:
            ax.legend()
            ax.set_ylabel(error_metric)
        else:
            ax.yaxis.set_ticks([])
        ax.set_xlabel(k)
        
        if error_metric is 'correlation':
            ax.set_ylim([0.4, 1.0])

    return fig, axes


def plot_subplots_complex(my_cmap, all_model_objs,
                          true_plot_dat, all_plot_dat, all_labels,
                          ind_var_between_plots, ind_var_in_plot,
                          plot_mode='eigs', # Dummy; must be eigs
                          error_metric='L2', latex_title=True):
    # Plots subplots with ONE EIGENVALUE and error circles around it

    subplot_key = list(ind_var_between_plots.keys())[0]
    subplot_vec = ind_var_between_plots[subplot_key]
    num_subplots = len(subplot_vec)
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots)
    
    k = list(ind_var_in_plot.keys())[0]
    num_columns = len(ind_var_in_plot[k])

    for i_subplot in range(num_subplots):
        unit_circle = tpu.get_circle()
        ax = axes[i_subplot]
        ax.add_artist(unit_circle)
        already_plotted_true = False
        for i_model in range(len(all_model_objs)):
            for i_x in range(num_columns):
                # Get eigs for each data instance
                e = all_plot_dat[i_model][i_subplot][i_x]
                if i_x == 0:
                    this_label = all_labels[i_model][i_subplot][0].replace(
                            '_', ';')
#                    this_label = this_label+
                else:
                    this_label = ''
                # Get the cloud around the FIRST eigenvalue
                # First means: imag()>0, and close to 0
                true_eigs = np.array(true_plot_dat[i_subplot][i_x])
                this_true_eig = None
                for t in true_eigs:
                    if t.imag < 0:
                        continue
                    if this_true_eig is None:
                        this_true_eig = t
                    elif t.imag < this_true_eig.imag:
                        this_true_eig = t
                # Plot the true values
                if not already_plotted_true:
                    ax.plot(this_true_eig.real, this_true_eig.imag,
                            'k+', ms=20.0, label='True')
                    already_plotted_true = True
                y = []
                for e_one_model in e:
                    ind = calc_eig_error(e_one_model, [this_true_eig],
                                         error_metric)[1]
                    y.append(e_one_model[ind])
                y = np.array(y)
                # Finally plot the data
                error_circle = tpu.get_circle(
                    center=(np.mean(y.real), np.mean(y.imag)),
                    r=np.std(y),  # Check
                    color=my_cmap[i_model])
                ax.add_artist(error_circle)
                ax.plot(y.real, y.imag, my_cmap[i_model]+'o', alpha=0.2,
                        label=this_label)
        if latex_title:
            ax.set_title("$\%s=%.2f$" % (subplot_key, subplot_vec[i_subplot]))
        else:
            ax.set_title("%s=%.2f" % (subplot_key, subplot_vec[i_subplot]))
        if i_subplot==0:
            ax.legend(loc='lower left')
            ax.set_ylabel('Imaginary part')
            common_ylim = list(ax.get_ylim())
            common_xlim = list(ax.get_xlim())
        else:
            ax.yaxis.set_ticks([])
            this_ylim = list(ax.get_ylim())
            common_ylim[0] = 0.9*min([common_ylim[0], this_ylim[0]])
            common_ylim[1] = 1.1*max([common_ylim[1], this_ylim[1]])
            this_xlim = list(ax.get_xlim())
            common_xlim[0] = min([common_xlim[0], this_xlim[0]])
            common_xlim[1] = max([common_xlim[1], this_xlim[1]])
        ax.set_xlabel('Real part')
    
    # Set all y axes to common limits
    for ax in axes:
        ax.set_ylim(common_ylim)
        ax.set_xlim(common_xlim)

    return fig, axes


def plot_subplots_histogram(my_cmap, all_model_objs,
                              true_plot_dat_list,
                              all_plot_dat_list,
                              all_labels_list,
                              ind_var_between_plots, 
                              ind_var_in_plot,
                              plot_mode='reconstruction',
                              error_metric='L2',
                              latex_title=True,
                              ind=None):
    # Plots subplots with different histograms

    subplot_key = list(ind_var_between_plots.keys())[0]
    subplot_vec = ind_var_between_plots[subplot_key]
    num_subplots = len(subplot_vec)
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots)

    k = list(ind_var_in_plot.keys())[0]
    num_columns = len(ind_var_in_plot[k])
    x_vec = ind_var_in_plot[k]

    for i_subplot in range(num_subplots):
        hist_dat = []
        for i_system in range(len(all_plot_dat_list)):
            all_labels = all_labels_list[i_system]
            all_plot_dat = all_plot_dat_list[i_system]
            true_plot_dat = true_plot_dat_list[i_system]
            ax = axes[i_subplot]
            # Calculate the data, then plot outside the loops
            tmp1, tmp2, tmp3 = calc_boxplot_data(all_model_objs,
                                                   all_labels,
                                                   all_plot_dat,
                                                   true_plot_dat,
                                                   error_metric,
                                                   num_columns,
                                                   plot_mode,
                                                   ind,
                                                   x_vec,
                                                   i_subplot)
            box_dat, tick_labels, group_labels = tmp1, tmp2, tmp3
            if len(box_dat[0][0]) == 0:
                if error_metric is 'correlation':
                    box_dat[0][0] = [0]
            this_improvement = np.mean(box_dat[1][0]) - np.mean(box_dat[0][0])
            hist_dat.append(this_improvement.real)
        
        # Finally plot
        plt.sca(ax)
        plt.hist(hist_dat)
        if latex_title:
            ax.set_title("$\%s=%.2f$" % (subplot_key, subplot_vec[i_subplot]))
        else:
            ax.set_title("%s=%.2f" % (subplot_key, subplot_vec[i_subplot]))
        if i_subplot==0:
            ax.set_ylabel("counts")
        else:
            ax.yaxis.set_ticks([])
        ax.set_xlabel("Improvement in %s" % error_metric)
        
    return fig, axes

###############################################################################
# Helper functions within the plotting functions...

def calc_boxplot_data(all_model_objs,
                      all_labels,
                      all_plot_dat,
                      true_plot_dat,
                      error_metric,
                      num_columns,
                      plot_mode,
                      ind,
                      x_vec,
                      i_subplot):
    
    box_dat, group_labels, tick_labels = [], [], []
    for i_model in range(len(all_model_objs)):
        model_dat = []
        for i_x in range(num_columns):
            # Get eigs or reconstruction for each data instance
            e = all_plot_dat[i_model][i_subplot][i_x]
            if i_x == 0:
                group_labels.append(
                        all_labels[i_model][i_subplot][0].replace(
                                '_', ';'))
            t = true_plot_dat[i_subplot][i_x]
            y = []
            for e_one_model in e:
                if plot_mode is 'eigs':
                    y_tmp = calc_eig_error(e_one_model, t, error_metric)[0]
                elif plot_mode is 'reconstructed_data':
                    y_tmp = calc_reconstruction_error(e_one_model,
                                                      t, error_metric,
                                                      ind=ind)
                else:
                    raise ValueError("Unknown plot mode")
                y.append(y_tmp)
            model_dat.append(np.array(y)[np.isfinite(y)])
            if i_model == 0:
                tick_labels.append('%d' % x_vec[i_x])
        box_dat.append(model_dat)
        
    return box_dat, tick_labels, group_labels
