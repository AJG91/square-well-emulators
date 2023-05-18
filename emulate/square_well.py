import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares


class SquareWell:
    """
    Defines a square well potential.
    
    Parameters
    ----------
    V0 : float
        Depth of potential (V0 > 0).
    R : float
        Radius of square well. R = 1 by default but explicit in class.
        
        
    Methods
    -------
    Vr(r)
        Returns a square well potential with radius r.
    E_gs()
        Function to find the lowest energy bound state.
    E_out(k)
        Returns the value of the energy outside the well at wave number k.
    k_in(E)
        Returns the value of the k vector inside the well at energy E.
    k_out(E)
        Returns the (imaginary) value of the k vector outside the well at energy E.
    kinetic()
        Returns the kinetic energy inside and outside the potential well.
    trans_eq(r)
        Transcendental function that is equal to zero when the square well
        has a bound state at r. Used for plotting. 
    f_E(E)
        Transcendental function that is equal to zero when the square well
        has a bound state at E.
    wf_in(r)
        Returns the unnormalized ground state wavefunction inside the well at radius r. 
    wf_out(r)
        Returns the unnormalized ground state wavefunction outside the well at radius r.
    total_wf(r)
        Returns the full normalized wave function across all points r inside and out.
    """
    
    def __init__(self, V0=1.0, R=1.0):
        self.V0 = V0
        self.R = R
        self.n = 1.0
        
        mu = 1.0
        hbar = 1.0
        self.hbar_sq_2M = hbar**2 / (2 * mu)
        
    def make_grid(self, r):
        # This separates points between inside and outside well with boundary r=R.
        # Makes sure to evenly space points out to avoid size errors.
        
        pts_in_pos = np.where(r < self.R)
        pts_in = r[pts_in_pos[0][0]:pts_in_pos[0][-1] + 1]
        
        pts_out_pos = np.where(r >= self.R)
        pts_out = r[pts_out_pos[0][0]:pts_out_pos[0][-1] + 1]
        
        return pts_in, pts_out
        
    def Vr(self, r, Vd=None):  
        """ 
        Potential for the square well.
        """   
        if (Vd is None):
            Vd = self.V0
        return np.where(r > self.R, 0., -Vd)  
    
    def E_gs(self):
        """
        Ground state energy inside well.
        """        
        E_guess = 0.95 * (-self.V0)
        xi = least_squares(self.f_E, E_guess)
        return xi.x[0]
    
    def E_out(self, k):   
        """ 
        Kinetic energy (outside well). 
        """
        return self.hbar_sq_2M * k**2
    
    def k_in(self, E):
        """
        Wave number inside well.
        """
        return np.sqrt((self.V0 + E) / self.hbar_sq_2M)
                     
    def k_out(self, E):
        """
        Wave number (real) outside well (negative energy)
        """
        return np.sqrt(-E / self.hbar_sq_2M)
    
    def kinetic(self):
        """
        Returns the kinetic energy. 
        """
        T_in = self.hbar_sq_2M * self.k_in(self.E_gs())**2
        T_out = -self.hbar_sq_2M * self.k_out(self.E_gs())**2
        
        return np.array([T_in, T_out])
    
    def f_E(self, E):
        """
        Function that is zero for E equal to a square-well bound-state energy.
        """
        xi = self.k_in(E) * self.R
        eta = self.k_out(E) * self.R
        return eta / xi  + 1 / np.tan(xi)
    
    def wf_in(self, r):
        """
        Returns the unnormalized wavefunction inside well.
        This returns U(r) inside the well where U(r) = r*R(r) 
        and R(r) is the actual radial wavefunction.
        """
        return np.sin(self.k_in(self.E_gs()) * r)
    
    def wf_out(self, r):
        """
        Returns the unnormalized wavefunction outside well.
        This returns U(r) outside the well where U(r) = r*R(r) 
        and R(r) is the actual radial wavefunction.
        """
        return np.exp(-self.k_out(self.E_gs()) * r)
    
    def k_out_scat(self, E):
        """
        Wave number (real) outside well (positive energy)
        """
        return np.sqrt(E / self.hbar_sq_2M)
    
    def phase_shift(self, k):
        """
        Phase shift (\delta_0) from continuity condition.
        """
        k1 = np.sqrt((self.E_out(k) + self.V0) / self.hbar_sq_2M)
        k2 = np.sqrt(self.E_out(k) / self.hbar_sq_2M)
        return -k2 * self.R + np.arctan((k2 / k1) * np.tan(k1 * self.R))
    
    def phase_shift_gen(self, k):
        """
        General equation for phase shift.
        \delta = - arctan(B/A); B = - C sin(\delta), A = C cos(\delta).
        """
        A = np.cos(self.phase_shift(k))
        B = -np.sin(self.phase_shift(k))
        return -np.arctan(B / A)
    
    def scat_amp(self, k):
        """
        Returns 1/f where f is the scattering amplitude.
        f = (1/k)*e^(i\delta_0)*sin(\delta_0).
        1/f = k \cot(\delta(k)) (leaving out imaginary part).
        """
        return k / np.tan(self.phase_shift(k))
    
    def wf_in_scat(self, r, E):
        """
        Returns the unnormalized scattering wavefunction inside well.
        This returns U(r) inside the well where U(r) = r*R(r) 
        and R(r) is the actual radial wavefunction.
        There is a distinction between the scattering and non-scattering
        wavefunction. The scattering wf can be at any energy, while the
        non-scattering wf has to be at the bound state energy.
        """
        return np.sin(self.k_in(E) * r)
    
    def wf_out_scat(self, r, E):
        """
        Returns the normalized scattering wavefunction outside well.
        This returns U(r) outside the well where U(r) = r*R(r) 
        and R(r) is the actual radial wavefunction.
        There is a distinction between the scattering and non-scattering
        wavefunction. The scattering wf can be at any energy, while the
        non-scattering wf has to be at the bound state energy.
        U(r) = e^(i \delta)/k2 * sin(k2*r + delta) 
             = sin(k2*r + delta)/(k2*cos(delta)) (real part only)
        """
        k2 = self.k_out_scat(E)
        delta = self.phase_shift_gen(k2)
        return np.sin(k2 * r + delta) / (k2 * np.cos(delta)) 
    
    def asymp_wf(self, r, E):
        """
        Returns the asymptotic form of the wave function for l=0. 
        Limit of the Bessel function j(kl) for large values of r.
        """
        return np.sin(self.k_out_scat(E) * r) / self.k_out_scat(E)
        
    def wf_coeff_scat(self, E):
        """
        Returns the normalization coefficient for the wavefunction inside well.
        Wave functions evaluated at boundary.
        """
        return self.wf_out_scat(self.R, E) / self.wf_in_scat(self.R, E)
    
    def wf_norm_scat(self, r, E):
        """
        Returns the complete normalized wavefunction.
        """
        pts_in, pts_out = self.make_grid(r)
        
        coeff = self.wf_coeff_scat(E)
        psi_in = coeff * self.wf_in_scat(pts_in, E)
        psi_out = self.wf_out_scat(pts_out, E)
        
        avg_val = (psi_in[-1] + psi_out[1]) / 2
        np.put(psi_out, psi_out[0], avg_val)
        psi_tot = np.append(psi_in, psi_out)
        
        return psi_tot
    
    def wf_tot(self, ps, ws):
        """
        Returns the full, normalized wavefunction across all points r.
        This returns U(r) inside and outside well where U(r) = r*R(r) 
        and R(r) is the actual wavefunction.
        """
        T = self.kinetic()
        T_in, T_out = T[0], T[1]
        pts_in, pts_out = self.make_grid(ps)
        
        ws_in, ws_out = ws[0:len(pts_in)], ws[len(pts_in):len(pts_in) + len(pts_out)]
                
        # Potential inside and outside
        V_in, V_out = self.Vr(pts_in), self.Vr(pts_out)
        
        # Wave functions inside and outside well
        wf_in, wf_out = self.wf_in(pts_in), self.wf_out(pts_out)
        
        # Boundary condition, comes from the continuity condition
        bc_coeff = self.wf_in(self.R) * self.wf_out(-self.R)
        
        # Calculate the normalization coefficient.       
        ans_wf1 = wf_in**2 @ ws_in
        ans_wf2 = bc_coeff**2 * wf_out**2 @ ws_out
        norm_coeff = np.sqrt(1 / (ans_wf1 + ans_wf2))
        
        # Expectation values of kinetic and potential energy
        T_ev = norm_coeff**2 * (wf_in**2 * T_in @ ws_in \
               + bc_coeff**2 * wf_out**2 * T_out @ ws_out)
                
        V_ev = norm_coeff**2 * (wf_in**2 * V_in @ ws_in \
               + bc_coeff**2 * wf_out**2 * V_out @ ws_out)
        
        self.T_ev, self.V_ev, self.E_ev = T_ev, V_ev, T_ev + V_ev
        
        # Normalized wave functions
        wf1_norm = norm_coeff * wf_in
        wf2_norm = norm_coeff * bc_coeff * wf_out

        # This smooths out the wavefunction at r=R by averaging between 
        # the last point of the inside wf with the second point on the outside wf.
        # First point of outside wf = last point of inside wf.
        avg_val = (wf1_norm[-1] + wf2_norm[1]) / 2
        np.put(wf2_norm, wf2_norm[0], avg_val)
        
        # Total wavefunction
        psi_tot = np.append(wf1_norm, wf2_norm)
        
        return psi_tot
    
    