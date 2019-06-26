# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:05:34 2019

@author: charl
"""

# Note: Importing these runs them!

import matplotlib.pyplot as plt

print('Creating DMD figures')
import dat_comparison_figures_large_system
import dat_comparison_figures_small_system

print('Creating DMDc figures')
import eig_comparison_figures_control
import dat_comparison_figures_control # Both large and small systems

print('Creating Parameter comparison figure')
import parameter_comparison_figure


plt.close('all')
