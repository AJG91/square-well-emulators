"""
Used to apply the KVP emulator to a potential for two-body scattering.
"""
import numpy as np
from scipy.linalg import orth
from numpy import (
    pad, zeros, array, squeeze, identity,
    ndarray, swapaxes, einsum, resize
)
from numpy.typing import ArrayLike
from typing import Union, Optional

class KVP_emulator_scattering:
    r"""
    Method
    ------
    Creates an emulator environment used to apply the KVP emulator 
    to a given potential for scattering.
    
    Parameters
    ----------
    k : array
        Array of k values at which we are predicting the phase shifts.
    ps : array
        The mesh points.
    ws : array
        The corresponding weights of the mesh points.
    """
    def __init__(
        self, 
        k: ArrayLike,
        ps: ArrayLike,
        ws: ArrayLike
    ):
        if (isinstance(k, ndarray) == False):
            k = array([k]) # Casts into array in case of scalar value
            
        self.len_k = len(k)
        self.k = k
        self.ps = ps
        self.ws = ws
        
        mu = 1.0
        hbar = 1.0
        self.coeff = (2.0 * mu / hbar**2)
    
    def _solve_inv(
        self, 
        U: ArrayLike, 
        tau: ArrayLike,
        h: ArrayLike,
    ) -> ArrayLike:
        r"""
        Method
        ------
        A choice for solving the U matrix and obtaining
        the emulator prediction. This method uses np.linalg.solve.

        Parameters
        ----------
        U : (basis size) x (basis size) array
            Matrix used to make EC predictions.
        tau : array
            An array of the taus from the basis.
        h : float
            Nugget used to regulate the basis collinearity.
        """
        from numpy.linalg import solve
        
        len_basis = tau.shape[1]
        I = resize(identity(len_basis), (len(U), len_basis, len_basis))
        
        U_tilde = U + U.swapaxes(1, 2) + h * I
        npad = ((0, 0), (0, 1), (0, 1))
        b = pad(tau, pad_width=npad[0:2], mode='constant', constant_values=1)
        A = pad(U_tilde, pad_width=npad, mode='constant', constant_values=1)
        A[:, -1, -1] = 0
               
        c_j = solve(A, b)[:, 0:len_basis]
        
        cj_tau = einsum('ij, ij -> i', c_j, tau, optimize=True)
        cj_U_cj = einsum('ij, ijk, ik -> i', c_j, U, c_j, optimize=True)
    
        return cj_tau - cj_U_cj, c_j
        
    def _leastsq_inv(
        self, 
        U: ArrayLike, 
        tau: ArrayLike,
        h: ArrayLike,
    ) -> ArrayLike:
        r"""
        Method
        ------
        A choice for solving the U matrix and obtaining
        the emulator prediction. This method uses np.linalg.lstsq.

        Parameters
        ----------
        U : (basis size) x (basis size) array
            Matrix used to make EC predictions.
        tau : array
            An array of the taus from the basis.
        h : ArrayLike
            List of nugget used to regulate the basis collinearity.
        """
        from scipy.linalg import lstsq
        
        len_k = self.len_k
        len_basis = tau.shape[1]
        c_j = zeros((len_k, len_basis))
        
        U_tilde = U + U.swapaxes(1, 2)
        
        npad = ((0, 0), (0, 1), (0, 1))
        b = pad(tau, pad_width=npad[0:2], mode='constant', constant_values=1)
        A = pad(U_tilde, pad_width=npad, mode='constant', constant_values=1)
        A[:, -1, -1] = 0
        
        for i, (A_i, b_i) in enumerate(zip(A, b)):
            c_j[i] = lstsq(A_i, b_i.T, cond=h)[0][0:len_basis]
        
        cj_tau = einsum('ij, ij -> i', c_j, tau, optimize=True)
        cj_U_cj = einsum('ij, ijk, ik -> i', c_j, U, c_j, optimize=True)
        
        return cj_tau - cj_U_cj, c_j
                
    def prediction(
        self, 
        tau: ArrayLike,
        h: float = 1e-12,
        sol_type: str = 'solve'
    ) -> ArrayLike:
        r"""
        Method
        ------
        Applies the KVP emulator to interpolate/extrapolate the 
        phase shifts for a specific scattering energy.

        Input
        -----
        tau : array
            The values of tau used to train the basis..
        h : float or list
            Nugget used to regulate the basis collinearity.
            If None, a random value within some range is chosen.
        sol_type : str
            Chooses what method to use to solve for the EC predictions.
            Choices: 'lstsq', 'solve', 'pinv'. 
            
        Output
        ------
        tau_var : float
            Returns the variational tau calculated using the emulator.
        c_var : float
            Returns the weights calculated using the emulator.
        """
        U = self.U_ij
        
        if sol_type == 'lstsq':
            tau_var, c_var = self._leastsq_inv(U, tau, h)
        elif sol_type == 'solve':
            tau_var, c_var = self._solve_inv(U, tau, h)
        else:
            raise Exception('Specify how to solve dU matrix!')
            
        if len(tau_var) == 1:
            tau_var = squeeze(tau_var)
            
        return tau_var, c_var
    
    def train(
        self, 
        wf_b: ArrayLike,
        V_b: ArrayLike,
        pot: ArrayLike
    ) -> ArrayLike:
        r"""
        Method
        ------
        Calculates the kernel matrix \delta \tilde{U}_{ij} in coordinate space.

        Input
        -----
        wf_b : array
            Wave functions used to train the basis. Calculated at r.
        V_b : array
            Potential used to train the basis. Calculated at r.
        pot : array
            Potential used for prediction.
            
        Output
        ------
        None
        """
        ws = self.ws
        len_k = self.len_k
        coeff = self.coeff
        
        U_ij = zeros((len_k, wf_b.shape[0], wf_b.shape[0]))
        
        wf_b_orth = orth(np.column_stack(wf_b)).T
        n_mat = wf_b @ wf_b.T
        
        for m in range(len_k):
            for i, wf_i in enumerate(wf_b):
                for j, (wf_j, V_j) in enumerate(zip(wf_b, V_b)):
                    U_ij[m][i][j] = coeff * (wf_i * (pot - V_j) * wf_j) @ ws
                
        self.U_ij = U_ij
        return None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    