"""
Derived module from dmdbase.py for optimized DMD, which is implemented in
MATLAB and Fortran.
Several packages must be installed for this to work, including of course 
MATLAB itself. Instructions are on:
    https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
"""
from .dmdbase import DMDBase
import matlab.engine
import numpy as np


class optDMD(DMDBase):
    """
    Optimized Dynamic Mode Decomposition

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    """
    
    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False):
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        self.original_time = None
        self.dmd_time = None

        self._eigs = None
        self._Atilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None
        
        self._eng = matlab.engine.start_matlab()

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        eng = self._eng
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        n_samples = self._snapshots.shape[1]
        
        # Cast into MATLAB types
        X = matlab.double(self._snapshots.tolist())
        tspan = matlab.double(list(range(n_samples)))
        r = self.svd_rank
        if r > 0:
            r = matlab.int16([r])
        elif r == -1:
            r = np.shape(X)[0]
        elif r == 0:
            U, s, V = np.linalg.svd(X, full_matrices=False)
            omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            r = np.sum(s > tau)
        i_mode = matlab.int16([2])
        
        t1, eigs, t3, t4, basis, A = eng.optdmd(X, tspan, r, i_mode, nargout=6)
        
        # optdmd returns continuous time eigenvalues
        self._eigs = np.array([np.exp(e) for e in eigs]).flatten()
        
        f = np.array
        self._modes, self._b, self._Atilde = f(t1), f(t3), f(t4)

        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        return self
    
    def clean_up(self):
        self._eng.quit()

    @property
    def label_for_plots(self):
        """Defines a name to be used in plotting"""
        if self.tlsq_rank > 0.0:
            return 'tls-opt-DMD'
        else:
            return 'opt-DMD'