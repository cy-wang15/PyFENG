import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import scipy.integrate as spint
import scipy.optimize as spop
import math
from scipy import interpolate
from scipy.misc import derivative
from pyfeng import sv_abc as sv

class HestonMcAe2(sv.SvABC, sv.CondMcBsmABC):
    """
    Almost exact MC for Heston model.

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.
    Example:
        >>> import numpy as np
        >>> import pyfeng.ex as pfex
        >>> strike = np.array([60, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pfex.HestonMcAe2(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_mc_params(n_path=1e4, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([12.08981758,  0.33379748, 42.28798189])  # not close so far
    """
    dist = 0

    def set_mc_params(self, n_path=10000, rn_seed=None, antithetic=True, dist=0):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            rn_seed: random number seed
            antithetic: antithetic
            dist: distribution to use for approximation. 0 for inverse Gaussian (default), 1 for lognormal.
        """
        self.n_path = int(n_path)
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.dist = dist

    def vol_paths(self, tobs):
        return np.ones(size=(len(tobs), self.n_path))

# define NCX variables
    def chi_dim(self):
        """
        Noncentral Chi-square (NCX) distribution's degree of freedom

        Returns:
            degree of freedom (scalar)
        """
        result = 4 * self.theta * self.mr / self.vov ** 2
        return result

    def chi_lambda(self, texp):
        """
        Noncentral Chi-square (NCX) distribution's noncentrality parameter

        Returns:
            noncentrality parameter (scalar)
        """
        chi_lambda = 4 * self.mr * np.exp(-self.mr * texp)/ (self.vov ** 2 *(1 - np.exp(-self.mr * texp))) * self.sigma
        return chi_lambda
    
# use for the BesselI function
    def nu(self):
        nu = 0.5 * self.chi_dim() -1
        return nu   
    
# define  var_t 
    def var_t(self, texp):

        chi_dim = self.chi_dim()
        chi_lambda = self.chi_lambda(texp)

        cof = self.vov ** 2 * (1 - np.exp(-self.mr * texp)) / (4 * self.mr)
        var_t = cof * np.random.noncentral_chisquare(chi_dim, chi_lambda, self.n_path)
        return var_t
    
#define MGF
#First define r(s) and its differention
    def r(self,s):
        result = np.sqrt(self.mr ** 2 + 2 * s * self.vov ** 2)
        return result

    def dr_ds(self,s):
        result = self.vov**2 / self.r(s)
        return result

#define BesselI function and its differention
    def I(self,nu,z):
        result = spsp.iv(nu,z)
        return result

    def dI_dz(self,nu,z):
        result = self.I(nu-1,z)-nu/z*self.I(nu,z)
        return result

    def d2I_dz2(self,nu,z):
        result = ( self.I(nu-2,z)+2*self.I(nu,z)+self.I(nu+2,z) ) / 4
        return result

#define subfunctions
    def f(self,s,texp,var_t):
        result =( ( self.sigma + var_t )/ self.vov ** 2 )  * s * ( 1 + np.exp(-s*texp))/( 1 - np.exp(-s*texp))
        return result

    def g(self,s,texp,var_t):
        print(s, texp)
        result = 4 * np.sqrt(self.sigma * var_t ) * s * np.exp(-0.5*s*texp)/(1 - np.exp(-s*texp))
        return result

    def df_ds(self,s,texp,var_t):
        result = ((self.sigma + var_t )/self.vov ** 2) * (1-2*texp*s*np.exp(-s*texp)-np.exp(-2*s*texp))/(1-np.exp(-s*texp))**2
        return result

    def dg_ds(self,s,texp,var_t):
        numerator = (1-0.5*s*texp)*np.exp(-0.5*s*texp)-(1+0.5*s*texp)*np.exp(-1.5*s*texp)
        denominator = (1-np.exp(-s*texp))**2
        result = 4 * np.sqrt(self.sigma * var_t) * numerator/ denominator
        return result

    def d2f_ds2(self,s,texp,var_t):
        result = 4*texp*np.exp(-s*texp) * ((1+0.5*s*texp)*np.exp(-s*texp)-1)/(1-np.exp(-s*texp))**3
        return result* ((self.sigma + var_t )/self.vov ** 2)

    def d2g_ds2(self,s,texp,var_t):
        numerator = (1.25*s*texp**2-3*texp)*np.exp(-0.5*s*texp)+(1.5*s*texp**2+4*texp)*np.exp(-1.5*s*texp)-(0.75*s*texp**2+texp)*np.exp(-2.5*s*texp)
        denominator = (1-np.exp(-s*texp))**3
        result = 4 * np.sqrt(self.sigma * var_t) * numerator/denominator
        return result

#define subfunctions
    def fai(self,s,texp,var_t):
        result = self.I(self.nu(),self.g(s,texp,var_t))/np.sinh(0.5*s*texp)/np.exp(self.f(s,texp,var_t))
        return result

# define dfai_ds/fai(first differention)
    def dfai_ds_fai(self,s,texp,var_t):
        result = (self.I(self.nu()-1,self.g(s,texp,var_t))/self.I(self.nu()-1,self.g(s,texp,var_t)) - self.nu()/self.g(s,texp,var_t)) * self.dg_ds(s,texp,var_t)-0.5*texp/np.tanh(0.5*s*texp)-self.df_ds(s,texp,var_t)
        return result

# define d2fai_ds2/fai(second differention)
    def d2fai_ds2_fai(self,s,texp,var_t):
        result_1 = - (self.dI_dz(self.nu(),self.g(s,texp,var_t))*self.dg_ds(s,texp,var_t))**2 / self.I(self.nu(),self.g(s,texp,var_t))**2
        result_2 =( self.d2I_dz2(self.nu(),self.g(s,texp,var_t))*self.dg_ds(s,texp,var_t) + self.dI_dz(self.nu(),self.g(s,texp,var_t))* self.d2g_ds2(s,texp,var_t) )/self.I(self.nu(),self.g(s,texp,var_t))
        result_3 = 0.25*texp**2*(1/np.sinh(0.5*s*texp)**2)
        result_4 = -self.d2f_ds2(s, texp, var_t)
        result_5 = result_1+result_2+result_3+result_4
        result = result_5 - self.dfai_ds_fai(s, texp, var_t)**2
        return result

#define MGF(-sX):
    def MGF(self,s,texp,var_t):
        result = self.fai(self.r(s),texp,var_t)/self.fai(self.mr,texp,var_t)
        return result

#find M1,M2:
    def moment_1(self,s,texp,var_t):
        result = self.dr_ds(s)*self.dfai_ds_fai(self.r(s),texp,var_t)*self.MGF(s,texp,var_t)
        return result

    def moment_2(self,s,texp,var_t):
        result = (self.vov**4/self.r(s)**3 )*self.MGF(self.r(s),texp,var_t)*(self.r(s)*self.d2fai_ds2_fai(self.r(s),texp,var_t)-self.dfai_ds_fai(self.r(s),texp,var_t))
        return result

    #find value
    def cond_spot_sigma(self, texp):
            var_t = self.var_t(texp)
            rhoc = np.sqrt(1.0 - self.rho ** 2)
            m1 = - self.moment_1(0,texp,var_t)
            m2 = self.moment_2(0,texp,var_t)

            if self.dist == 0:
                scale_ig = m1 ** 3 / (m2 - m1 ** 2)
                miu_ig = m1 / scale_ig
                int_var_std = spst.invgauss.rvs(miu_ig, scale=scale_ig) / texp
            elif self.dist == 1:
                scale_ln = np.sqrt(np.log(m2) - 2 * np.log(m1))
                miu_ln = np.log(m1) - 0.5 * scale_ln ** 2
                int_var_std = np.random.lognormal(miu_ln, scale_ln) / texp
            else:
                raise ValueError(f"Incorrect distribution.")

            ### Common Part
            int_var_dw = ((var_t - self.sigma) - self.mr * texp * (self.theta - int_var_std)) / self.vov
            spot_cond = np.exp(self.rho * (int_var_dw - 0.5 * self.rho * int_var_std * texp))
            sigma_cond = rhoc * np.sqrt(int_var_std / self.sigma)  # normalize by initial variance

            # return normalized forward and volatility
            return spot_cond, sigma_cond
