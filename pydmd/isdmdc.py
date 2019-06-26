"""
Derived module from dmdbase.py for infinite series dmd.

Author: Charles Fieseler
"""
import numpy as np

from .dmdbase import DMDBase
from past.utils import old_div
from math import factorial, pi, floor, isnan
from pydmd import DMDc
import warnings


class iDMDc(DMDBase):
    """
    Infinite series DMD class, with control. Note that although the base class
    allows for multiple types of series, only the geometric series is feasible
    here.

    Implemented:
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
                 lambda_factor=1.0, truncation=None,
                 settings_from_obj=None, adaptive_tol=1e-8,
                 to_fix_aliasing=False):
        if settings_from_obj is None:
            self.svd_rank = svd_rank
            self.tlsq_rank = tlsq_rank
            self.exact = exact
            self.opt = opt
            # New for this method
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
            
        if self.lambda_factor is None and self.truncation is not None:
            self.calc_lambda()
        
        self.original_time = None
        self.dmd_time = None
        
        self._eigs = None
        self._Atilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._A = None
        self._B = None
        self._snapshots = None
        self._snapshots_shape = None
        self._controlin = None
        self._controlin_shape = None
        self._Y = None
        self._max_pow = None
        
    @property
    def B(self):
        """
        Get the operator B.

        :return: the operator B.
        :rtype: numpy.ndarray
        """
        return self._B
    
    def reconstructed_data(self, control_input=None):
        """
        Return the reconstructed data, computed using the `control_input`
        argument. If the `control_input` is not passed, the original input (in
        the `fit` method) is used. The input dimension has to be consistent
        with the dynamics.

        :param numpy.ndarray control_input: the input control matrix.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        if control_input is None:
            controlin, controlin_shape = self._controlin, self._controlin_shape
        else:
            controlin, controlin_shape = self._col_major_2darray(control_input)

        if controlin.shape[1] != self.dynamics.shape[1]-1:
            raise RuntimeError(
                    'The number of control inputs and the number of snapshots to reconstruct has to be the same')

#        omega = old_div(np.log(self.eigs), self.original_time['dt'])
#        eigs = np.exp(omega * self.dmd_time['dt'])
#        A = self.modes.dot(np.diag(eigs)).dot(np.linalg.pinv(self.modes))
        A = self._A

        data = [self._snapshots[:, 0]]

        for i, u in enumerate(controlin.T):
            data.append(A.dot(data[i]) + self._B.dot(u))

        data = np.array(data).T
        return data

    def fit(self, X, I):
        """
        Compute the Dynamics Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        self._controlin, self._controlin_shape = self._col_major_2darray(I)

        n_samples = self._snapshots.shape[1]
        if self.to_fix_aliasing:
            max_pow = self.check_aliasing()
            lam, trunc = self.calc_adaptive_truncation(max_pow)
        else:
            lam, trunc = self.lambda_factor, self.truncation

        X = self._snapshots[:, :-trunc]
        Y = self.build_series_term('Y')
        ctr = self.build_series_term('U')

        self._fit_B_unknown(X, Y, ctr)

        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        return self

    def _fit_B_unknown_SVD(self, X, Y, ctr):
        """
        Private method that performs the dynamic mode decomposition algorithm
        with control when the matrix `B` is not provided.

        :param numpy.ndarray X: the first matrix of original snapshots.
        :param numpy.ndarray Y: the second matrix of original snapshots.
        :param numpy.ndarray I: the input control matrix.
        """

        lam = self.lambda_factor
        omega = np.vstack([X, ctr])

        Up, sp, Vp = self._compute_svd(omega, self.svd_rank)

        Up1 = Up[:self._snapshots.shape[0], :]
        Up2 = Up[self._snapshots.shape[0]:, :]
        # TODO: a second svd_rank?
        Ur, sr, Vr = self._compute_svd(Y, -1)
        self._basis = Ur

        Atilde = Ur.T.conj().dot(Y).dot(Vp).dot(
            np.diag(np.reciprocal(sp))).dot(Up1.T.conj()).dot(Ur)
        Btilde = Ur.T.conj().dot(Y).dot(Vp).dot(
            np.diag(np.reciprocal(sp))).dot(Up2.T.conj())
        
        self._Atilde = self._data_space_from_series(Atilde, lam)
        self._Btilde = self._data_space_from_series(Btilde, lam, Atilde)

        self._B = Ur.dot(self._Btilde)
        self._eigs, modes = np.linalg.eig(self._Atilde)
        self._modes = Y.dot(Vp).dot(np.diag(np.reciprocal(sp))).dot(
            Up1.T.conj()).dot(Ur).dot(modes)

        self._b = self._compute_amplitudes(self._modes, X, self._eigs, self.opt)
        
    def _fit_B_unknown(self, X, Y, ctr):
        """
        Private method that performs the dynamic mode decomposition algorithm
        with control when the matrix `B` is not provided VIA INVERSES, NOT SVD.

        :param numpy.ndarray X: the first matrix of original snapshots.
        :param numpy.ndarray Y: the second matrix of original snapshots.
        :param numpy.ndarray I: the input control matrix.
        """
        lam = self.lambda_factor
        omega = np.vstack([X, ctr])

        A_and_B = Y @ np.linalg.pinv(omega)
        Atilde = A_and_B[:, :self._snapshots.shape[0]]
        Btilde = A_and_B[:, self._snapshots.shape[0]:]
        
        self._Atilde = self._data_space_from_series(Atilde, lam)
        self._B = self._data_space_from_series(self._Atilde, lam, Btilde)
        self._A = self._Atilde

        # Complete the calculation of modes
#        U1, s1, V1 = self._compute_svd(self._Atilde, self.svd_rank)
#        A_lowrank = U1 @ np.diag(s1) @ V1
#        self._eigs, modes = np.linalg.eig(A_lowrank)
#        self._eigs, modes = np.linalg.eig(self._Atilde)
#        modes = modes[:self.svd_rank, :]
#        U, s, V = self._compute_svd(X, self.svd_rank)
#        self._modes = Y.dot(V).dot(np.diag(np.reciprocal(s))).dot(
#            U.T.conj()).dot(U).dot(modes)

        # Take two
        self._eigs, modes = np.linalg.eig(self._Atilde)
        Up, sp, Vp = self._compute_svd(omega, self.svd_rank)
        Ur, sr, Vr = self._compute_svd(Y, -1)
        Up1 = Up[:self._snapshots.shape[0], :]
        self._modes = Y.dot(Vp).dot(np.diag(np.reciprocal(sp))).dot(
            Up1.T.conj()).dot(Ur).dot(modes)

        self._b = self._compute_amplitudes(self._modes, X,
                                           self._eigs, self.opt)

    def check_aliasing(self):
        """
        Check to see if higher powers of the dynamics matrix will produce
        aliasing problems due to rotations of the eigenvalues in the complex
        plane of more than 2pi
        """
        naive_model = DMDc(svd_rank=-1)
        naive_model.fit(self._snapshots, self._controlin)
        eigs = naive_model._eigs
        max_pow = [(2*pi/e if e>0 else float('nan')) for e
                   in np.abs(np.angle(eigs))]
#        max_pow = np.abs([2*pi/(e+1e-6) for e in np.angle(eigs)])
#        assert(not all([isnan(e) for e in max_pow]),
#               "No maximum truncation could be determined")
        if isnan(min(max_pow)):
            warnings.warn("All eigenvalues are real, using default truncation")
            max_pow = [10]
        self._max_pow = max_pow

        if self.truncation is None:
            return max_pow
        elif self.truncation > floor(min(max_pow)):
            warnings.warn("Truncation is larger than the maximum " +
                          "non-aliasing value, %d" % floor(min(max_pow)))
            return max_pow

    def calc_adaptive_truncation(self, max_pow):
        """
        Calculates a truncation value adaptively that prevent aliasing, which
        happens because of eigenvalues passing the 2pi mark for higher matrix
        powers.
        """
        lam = self.lambda_factor
        trunc = self.truncation
        if trunc is None:
            assert(lam is None, "Lambda factor will be overwritten; " + 
                   "use adaptive_tol option instead")
            trunc = floor(min(max_pow))
            while True:
                err = self.calc_series_coef(lam, trunc+1)
                if err < self.adaptive_tol:
                    break
                else:
                    lam *= 0.75
            self.lambda_factor = lam
            self.truncation = trunc
            
        return lam, trunc
    

    def calc_lambda(self):
        """
        Calculates a lambda value given a nonzero truncation in order to keep
        the truncation error below the adaptive_tol value
        """
        self.lambda_factor = pow(self.adaptive_tol, 1.0/self.truncation)
    

    def build_series_term(self, func_mode):
        """
        Builds the finite truncation of the infinite series term, which will
        be both the data and the control term, as determined by "func_mode"
        """
        
        lam, trunc = self.lambda_factor, self.truncation
        
        if func_mode is 'Y':
            dat = self._snapshots
            if (trunc-1) > 0:
                Y = lam*dat[:, 1:-(trunc-1)]
            else:
                Y = lam*dat[:, 1:]
            offset = 0
        elif func_mode is 'U':
            dat = self._controlin
            if (trunc-1) > 0:
                Y = lam*dat[:, 0:-(trunc-1)]
            else:
                Y = lam*dat[:, 0:]
            offset = -1
            
        n, m = np.shape(self._snapshots)
        
        # Build the Y matrix; X is the same as DMD but slightly smaller
        for i in range(2, trunc+1):
            c = self.calc_series_coef(lam, i)
            ind = range(i + offset, m-trunc+i + offset)
            Y += c * dat[:, ind]
        
        return Y
    
    @staticmethod
    def calc_series_coef(lambda_factor, i):
        """Calculates geometric coefficients depending on series_type"""
        return lambda_factor**i

    @staticmethod
    def _data_space_from_series(Atilde, lam, G=None):
        """
        Converts back into the data space from a (truncated) infinite series,
        either exponential or power

        :param Atilde: the A matrix in the series-data space
        :param lam: the lambda multiplicative factor
        :param G: if passed, means that it will calculate the control matrix
        """
        identity = np.identity(np.shape(Atilde)[1])

        if G is None:
            A = Atilde @ np.linalg.inv(Atilde+identity) / lam
        else:
            A = (identity - lam*Atilde) @ G

#        return A, eigvals, eigvecs
        return A

#    def copy(self):
#        """
#        Copies the iDMD object settings to a new object
#        """
#        new_obj = iDMD(settings_from_obj=self)
#
#        return new_obj

    @property
    def label_for_plots(self, simple_name=True):
        """ Defines a name to be used in plotting"""
        name = 'iDMDc'
        if self.tlsq_rank > 0.0:
            name = 'tls-' + name
        if not simple_name and self.lambda_factor != 1.0:
            name = 'lam%.1f-' % self.lambda_factor + name
        return name
      