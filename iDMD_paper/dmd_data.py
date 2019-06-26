# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:42:36 2019

@author: charl
"""

import numpy as np
import random
from math import sin, cos, pi

class DMDData(object):
    """
    A class for producing DMD data, i.e. data produced by a linear dynamical
    system. Optionally, an external control can also be produced
    """

    def __init__(self,
                 n_A=2, arg_bounds=[1.0, 1.0], phase_bounds=[0.0, pi/4],
                 n_B=0, dim_increase=0, U_sparsity=0.99,
                 sigma=0.0,
                 A=None, B=None, U=None, C=None,
                 num_snapshots=1028,
                 num_datasets=10):
        # For later data production
        self.sigma = sigma
        self.num_datasets = num_datasets
        # For basic dynamics
        self.n_A = n_A
        self.arg_bounds = arg_bounds
        self.phase_bounds = phase_bounds
        self.num_snapshots = num_snapshots  # Does not limit data production
        # For controlled dynamics (optional)
        self.n_B = n_B
        self.dim_increase = dim_increase
        self.U_sparsity = U_sparsity
        # Actual matrices
        self.A, self.B, self.U = A, B, U
        self.C = C
        # Matrix properties
        self.eigs = None
        # Build them if they aren't passed
        if A is None:
            self.build_A_matrix()
            if n_B > 0:
                if B is None:
                    self.build_B_matrix()
                if U is None:
                    self.build_U_matrix(num_snapshots)
            if dim_increase > 0:
                self.build_C_matrix()

    ###########################################################################
    # Building dynamics matrices         
    def build_A_matrix(self, real_matrix=True):
        """
        Creates a random linear dynamic matrix (A) with bounded eigenvalues
        """
        n = self.n_A
        phase_bounds = self.phase_bounds
        arg_bounds = self.arg_bounds

        m = np.random.rand(n, n)
        vecs = np.linalg.eig(m-m.T)[1]  # Complex eigenvalues

        if real_matrix:
            assert n % 2 == 0
            n = int(n/2)

        eigs = []
        for i in range(n):
            theta = random.uniform(phase_bounds[0], phase_bounds[1])
            sign = 2*random.randint(0, 1)-1
            r = random.uniform(arg_bounds[0], arg_bounds[1])
            this_eig = complex(r*cos(theta), sign*r*sin(theta))
            eigs.append(this_eig)
            if real_matrix:
                eigs.append(np.conj(this_eig))

        A = vecs @ np.diag(eigs) @ np.linalg.inv(vecs)
        if real_matrix:
            A = A.real

        self.A = A
        self.eigs = eigs

    def build_B_matrix(self):
        """
        Builds a B matrix, currently just random.
        TODO: make it just a rotation?
        """
        self.B = np.random.rand(self.n_A, self.n_B)
        
    def build_U_matrix(self, num_snapshots):
        """
        Builds a sparse control signal, U
        """
        U = np.random.rand(self.n_B, num_snapshots)
        U[U<self.U_sparsity] = 0
        self.U = U
        return U
    
    def build_C_matrix(self):
        """
        Builds a measurement matrix, which increases the dimensionality of the
        system without increasing the rank
        """
        self.C = np.random.rand(self.n_A+self.dim_increase, self.n_A)

    ###########################################################################
    # Producing data (does NOT modify self)
    def produce_data(self, num_snapshots=None, x0=None, sigma=None):
        """
        Evolves the linear system forward in time, adds noise, and lastly
        applies the measurement matrix 
        
        Returns:
            X_noise - the data with noise, in the measurement basis
            X- the data without noise
        """
        if sigma is None:
            sigma = self.sigma
        m = num_snapshots
        if m is None:
            m = self.num_snapshots
        n_A = self.n_A
        A = self.A
        B = self.B
        if m > self.num_snapshots:
            print('Producing new control signal')
            U = self.build_U_matrix(num_snapshots)
        else:
            U = self.U
        
        noise = np.random.normal(0.0, sigma, m)  # gaussian noise
        X = np.zeros((n_A, m))
        X_noise = np.zeros((n_A, m))
        if x0 is None:
            X[:, 0] = np.random.rand(n_A)
        else:
            X[:, 0] = x0
        X_noise[:, 0] = X[:, 0] + noise[0]
        # evolve the system and perturb the data with noise
        if self.B is None:
            for k in range(1, m):
                X[:, k] = A @ X[:, k-1]
                X_noise[:, k] = X[:, k] + noise[k]
        else:
            for k in range(1, m):
                X[:, k] = A @ X[:, k-1] + B @ U[:, k-1]
                X_noise[:, k] = X[:, k] + noise[k]
        if self.dim_increase > 0:
            X_noise = self.C @ X_noise
            X = self.C @ X

        return X_noise, X
    
    
    def produce_data_cloud(self, num_snapshots=None, x0=None, sigma=None,
                           num_datasets=None):
        # Produces a list of datasets using produce_data()
        if num_datasets is None:
            num_datasets = self.num_datasets
        
        all_data = []
        for i in range(num_datasets):
            X_noise, X = self.produce_data(num_snapshots=num_snapshots,
                                           x0=x0,
                                           sigma=sigma)
            all_data.append(X_noise)
        
        return all_data, X
            