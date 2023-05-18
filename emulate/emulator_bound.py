"""
Used to apply the KVP emulator to a potential for bound states.
"""
import numpy as np
from numpy import(
    where, zeros, argsort, real, column_stack
)
from scipy.linalg import eig, orth

class KVP_emulator_bound:
    """
    Creates an emulator environment used to apply the KVP emulator 
    to a given potential for bound states.
    
    Parameters
    ----------
    ps : ArrayLike
    ws : ArrayLike
    V_pred : ArrayLike
    R : float
        
    Methods
    -------
    build_matrices
        Calculates the normalization and Hamiltonian matrix using the 
        wavefunction definition in the class SquareWell.
    energy_pred
        Calculates the eigenvalues of the Hamiltonian and the normalization matrix.
    """
    
    def __init__(self, ps, ws, V_pred, R=1):
        
        self.R = R
        self.V_pred = column_stack(V_pred).T
        pts_in, pts_out = self.make_grid(ps)
        
        self.ws = ws
        self.pts_in, self.pts_out = pts_in, pts_out
        
    def make_grid(self, r):
        """
        This separates points between inside and outside well with boundary r=R.
        Makes sure to evenly space points out to avoid size errors.
        """
        
        pts_in_pos = where(r < self.R)
        pts_in = r[pts_in_pos[0][0]:pts_in_pos[0][-1] + 1]
        
        pts_out_pos = where(r >= self.R)
        pts_out = r[pts_out_pos[0][0]:pts_out_pos[0][-1] + 1]
        
        return pts_in, pts_out
    
    def build_matrices(self, wf_b, T_b):
        """
        Builds the Hamiltonian matrix H and the N matrix.
        """
        ws, V_pred = self.ws, self.V_pred
        pts_in, pts_out = self.pts_in, self.pts_out
        
        len_basis = wf_b.shape[0]
        N = zeros((len_basis, len_basis))
        H = zeros((len_basis, len_basis))
        
        T_b_stack = column_stack(T_b)
        wf_stack = column_stack(np.sqrt(ws) * wf_b)
        
        N = wf_stack.T @ wf_stack
        V = wf_stack.T @ (V_pred * wf_stack)
    
        T_in = T_b_stack[0] * (wf_stack[:len(pts_in)].T @ wf_stack[:len(pts_in)])
        T_out = T_b_stack[1] * (wf_stack[len(pts_in):].T @ wf_stack[len(pts_in):])
        H = T_in + T_out + V
        
        return N, H

    def energy_pred(self, wf_b, T_b):
        """
        Calculates the eigenvalues \lambda for |H - \lambda N| = 0.
        """
        N, H = self.build_matrices(wf_b, T_b)
        
        w, v = eig(H, N)
        s = argsort(real(w))
        
        return real(w[s[0]])
    
    
    
    
    
    
    
    
    
    
    
    
    
    