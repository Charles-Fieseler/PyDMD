"""
Derived module from dmdbase.py for infinite series dmd.

Author: Charles Fieseler
"""
import numpy as np

from .dmdbase import DMDBase
from math import factorial, pi, floor, isnan
from pydmd import DMD
import warnings


class iDMD(DMDBase):
    """
    Infinite series DMD class.

    Implemented:
    :param str series_type: Either exponential or power series currently
    :param float lambda_factor: Multiplies each term in the series to change
        convergence properties; sometimes necessary for convergence
    :param int truncation: The number of steps to include in the series. As
        this is increased, the number of timesteps used for each term in the
        series decreases, i.e. each is (m-truncation)

    TO CHECK:
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive integer, the
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

    Reference: TODO
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False,
                 series_type='exponential', lambda_factor=1.0, truncation=None,
                 settings_from_obj=None, adaptive_tol=1e-8,
                 to_fix_aliasing=False):
        if settings_from_obj is None:
            self.svd_rank = svd_rank
            self.tlsq_rank = tlsq_rank
            self.exact = exact
            self.opt = opt
            # New for this method
            self.series_type = series_type
            self.lambda_factor = lambda_factor
            self.truncation = truncation  # Default is adaptive truncation
            self.adaptive_tol = adaptive_tol
            self.to_fix_aliasing = to_fix_aliasing
        else:
            self.svd_rank = settings_from_obj.svd_rank
            self.tlsq_rank = settings_from_obj.tlsq_rank
            self.exact = settings_from_obj.exact
            self.opt = settings_from_obj.opt
            # New for this method
            self.series_type = settings_from_obj.series_type
            self.lambda_factor = settings_from_obj.lambda_factor
            self.truncation = settings_from_obj.truncation
            self.adaptive_tol = settings_from_obj.adaptive_tol
            self.to_fix_aliasing = settings_from_obj.to_fix_aliasing
            
        self.original_time = None
        self.dmd_time = None
        
        self._eigs = None
        self._Atilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None
        self._Y = None
        self._max_pow = None

    def fit(self, X):
        """
        Compute the Dynamics Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        if self.to_fix_aliasing:
            max_pow = self.check_aliasing()
            lam, trunc = self.calc_adaptive_truncation(max_pow)
        else:
            lam, trunc = self.lambda_factor, self.truncation
        
        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-trunc]
        dat = self._snapshots
        Y = lam*dat[:, 1:-(trunc-1)]
        n, m = np.shape(self._snapshots)
        
        # Build the Y matrix; X is the same as DMD but slightly smaller
        for i in range(2, trunc+1):
            c = self.calc_series_coef(self.series_type, lam, i)
            ind = range(i, m-trunc+i)
            Y += c * dat[:, ind]

        # Do svd truncation on these like normal; is in a new space
        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)
        U, s, V = self._compute_svd(X, self.svd_rank)
        Atilde = self._build_lowrank_op(U, s, V, Y)
        self._Y = Y

        # Convert back to original data space
#        self._Atilde, self._eigs, self._modes = self._data_space_from_series(
#                Atilde, lam, self.series_type)
        self._Atilde = self._data_space_from_series(Atilde,
                                                    lam, self.series_type)

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, Y, U, s, V, exact=False)
        
        if self.to_fix_aliasing:
            self.fix_aliasing()

        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        return self

    def check_aliasing(self):
        """
        Check to see if higher powers of the dynamics matrix will produce
        aliasing problems due to rotations of the eigenvalues in the complex
        plane of more than 2pi
        """
        naive_model = DMD(svd_rank=-1)
        naive_model.fit(self._snapshots)
        eigs = naive_model._eigs
        max_pow = [(2*pi/e if e>0 else float('nan')) for e 
                   in np.abs(np.angle(eigs))]
#        max_pow = np.abs([2*pi/(e+1e-6) for e in np.angle(eigs)])
#        assert(not all([isnan(e) for e in max_pow]),
#               "No maximum truncation could be determined")
        if all([isnan(e) for e in max_pow]):
            warnings.warn("All eigenvalues are real, using default truncation")
            max_pow = [10]
            
        self._max_pow = max_pow

        if self.truncation is None:
            return max_pow
        elif self.truncation > floor(np.nanmin(max_pow)):
            warnings.warn("Truncation is larger than the maximum non-aliasing" +
                    " value, %d" % floor(np.nanmin(max_pow)))
            return max_pow
        
#    def fix_aliasing(self):
#        """
#        Fix aliasing due to large powers of the dynamics matrix A.
#        """
#        raw_coef = []
#        trunc = list(range(self.truncation))
#        for i in trunc:
#            raw_coef.append(self.calc_series_coef(self.series_type,
#                                                  self.lambda_factor, i))
#        # Do this process a couple times
#        all_angles = np.angle(self._eigs)
#        for i in range(4):
#            for ang in all_angles:
#                true_angles = ang*trunc
#                aliased_angles = [a % (2*pi) for a in true_angles]


    def calc_adaptive_truncation(self, max_pow):
        
        lam = self.lambda_factor
        trunc = self.truncation
        if trunc is None:
            assert(lam is None, "Lambda factor will be overwritten; " + 
                   "use adaptive_tol option instead")
            trunc = floor(np.nanmin(max_pow))
            while True:
                err = self.calc_series_coef(self.series_type, lam, trunc+1)
                if err < self.adaptive_tol:
                    break
                else:
                    lam *= 0.75
            self.lambda_factor = lam
            self.truncation = trunc
            
        return lam, trunc
    
    @staticmethod
    def calc_series_coef(series_type, lambda_factor, i):
        """Calculates series coefficients depending on series_type"""
        if series_type is 'exponential':
            return (lambda_factor**i) / factorial(i)
        elif series_type is 'geometric':
            return lambda_factor**i

    @staticmethod
    def _data_space_from_series(Atilde, lam, series_type):
        """
        Converts back into the data space from a (truncated) infinite series,
        either exponential or power

        :param Atilde: the A matrix in the series-data space
        :param lam: the lambda multiplicative factor
        :param series_type: the type of infinite series: exponential or power
        """
        identity = np.identity(np.shape(Atilde)[1])

        if series_type is 'exponential':
            eigvals, eigvecs = np.linalg.eig(Atilde + identity)
            eigvals = np.log(eigvals) / lam
#            eigvals, eigvecs = np.linalg.eig(Atilde / lam + identity)
            eigvals = np.log(eigvals)
            A = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
        elif series_type is 'geometric':
            A = Atilde @ np.linalg.inv(Atilde+identity) / lam
#            eigvals, eigvecs = np.linalg.eig(A)

#        return A, eigvals, eigvecs
        return A

    def copy(self):
        """
        Copies the iDMD object settings to a new object
        """
        new_obj = iDMD(settings_from_obj=self)

        return new_obj

    @property
    def label_for_plots(self, simple_name=True):
        """ Defines a name to be used in plotting"""
        name = 'iDMD'
        if self.tlsq_rank > 0.0:
            name = 'tls-' + name
        if not simple_name:
            if self.series_type is 'exponential':
                name = 'exp-' + name
            elif self.series_type is 'geometric':
                name = 'geo-' + name
            if self.lambda_factor != 1.0:
                name = 'lam%.1f-' % self.lambda_factor + name
        return name
            