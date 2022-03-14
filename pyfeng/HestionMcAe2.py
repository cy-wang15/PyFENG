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
    dist = 1

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
#define BesselI function and its differention
    def I(self,nu,z):
        result = spsp.iv(nu,z)
        return result

    def dI(self,nu,z):
        result = self.I(nu-1,z)-nu/z*self.I(nu,z)
        return result

    def d2I(self,nu,z):
        result = ( self.I(nu-2,z)+2*self.I(nu,z)+self.I(nu+2,z) ) / 4
        return result

#First define r(s) and its differention
    def r(self,s):
        result = np.sqrt(self.mr ** 2 + 2 * s * self.vov ** 2)
        return result

    def dr(self,s):
        result = self.vov**2 / self.r(s)
        return result
    
    def d2r(self,s):
        result = - self.vov**4/ self.r(s)**3
        return result
    
    def u(self,x,texp):
        result = 0.5*x*texp
        return result
    
    def du(self,x,texp):
        result = 0.5* texp
        return result
    
    def d2u(self,x,texp):
        return 0
    
#define subfunctions: A,f,g
    def A(self,u):
        result = u/np.sinh(u)
        return result
    
    def f(self,u,texp,var_t):
        cof = 2*(self.sigma + var_t )/ self.vov ** 2/texp
        result = cof  *  np.cosh(u) * self.A(u)
        return result

    def g(self,u,texp,var_t):
        cog = 4 * np.sqrt(self.sigma * var_t )/self.vov ** 2/texp
        result = cog *  self.A(u)
        return result
#define subfunctions
    def fai(self,u,texp,var_t):
        nu = self.nu()
        f = self.f(u,texp,var_t)
        g = self.g(u,texp,var_t)
        I = self.I(nu,g)
        A = self.A(u)
        result = I*np.exp(f)*A
        return result
#define MGF(-sX):   
    def MGF(self,s,texp,var_t):
        u_s = self.u(self.r(s))
        u_mr = self.u(self.mr)
        result = self.fai(u_s,texp,var_t)/self.fai(u_mr,texp,var_t)
        return result 
    
    
    
#define subfunctions differention:( dA,df,dg )/du
    def dA(self,u):
        result = (-np.cosh(u)*u+np.sinh(u))/np.sinh(u)**2
        return result
    
    def df(self,u,texp,var_t):
        cof = 2*(self.sigma + var_t )/ self.vov ** 2/texp
        result =cof * ( np.sinh(u)*self.A(u) + np.cosh(u)*self.dA(u) )
        return result

    def dg(self,u,texp,var_t):
        cog = 4 * np.sqrt(self.sigma * var_t )/self.vov ** 2/texp
        result =  cog * self.dA(u)
        return result
#define subfunctions:( d2A,d2f,d2g )/du2
    def d2A(self,u):
        result =(- u*np.sinh(u)**2 - 2*np.cosh(u)*np.sinh(u)+2*u*np.cosh(u)**2)/np.sinh(u)**3
        return result
    
    def d2f(self,u,texp,var_t):
        cof = 2*(self.sigma + var_t )/ self.vov ** 2/texp
        A = self.A(u)
        dA = self.dA(u)
        d2A = self.d2A(u)
        result = cof * (np.cosh(u)*(A+d2A)+2*np.sinh(u)*dA)
        return result

    def d2g(self,u,texp,var_t):
        cog = 4 * np.sqrt(self.sigma * var_t )/self.vov ** 2/texp
        result = cog* self.d2A(u)
        return result

#define subfunctions
    def fai(self,u,texp,var_t):
        nu = self.nu()
        f = self.f(u,texp,var_t)
        g = self.g(u,texp,var_t)
        I = self.I(nu,g)
        A = self.A(u)
        result = I*np.exp(-f)*A
        return result
#define MGF(-sX):   
    def MGF(self,s,texp,var_t):
        u_s = self.u(self.r(s))
        u_mr = self.u(self.mr)
        result = self.fai(u_s,texp,var_t)/self.fai(u_mr,texp,var_t)
        return result
    
# define dfai_du(first differention)
    def dfai(self,u,texp,var_t):
        nu = self.nu()
        f = self.f(u,texp,var_t)
        df = self.df(u,texp,var_t)
        g = self.g(u,texp,var_t)
        dg = self.dg(u,texp,var_t)
        I = self.I(nu,g)
        dI = self.dI(nu,g) * dg
        A = self.A(u)
        dA = self.dA(u)
    
        result = dI*np.exp(-f)*A + I*(-df)*np.exp(-f)*A +I*np.exp(-f)*dA
        return result
    
    def d2fai(self,u,texp,var_t):
        nu = self.nu()
        f = self.f(u,texp,var_t)
        df = self.df(u,texp,var_t)
        d2f = self.d2f(u,texp,var_t)
        g = self.g(u,texp,var_t)
        dg = self.dg(u,texp,var_t)
        d2g = self.d2g(u,texp,var_t)
        I = self.I(nu,g)
        dI = self.dI(nu,g) * dg
        d2I = self.d2I(nu,g) * dg**2 + self.dI(nu,g) * d2g
        A = self.A(u)
        dA = self.dA(u)
        d2A = self.d2A(u)
        
        result = d2I*np.exp(-f)*A + dI*(-df)*np.exp(-f)*A +dI*np.exp(-f)*dA + \
                dI*(-df)*np.exp(-f)*A + I*(-d2f+df**2)*np.exp(-f)*A+I*(-df)*np.exp(-f)*dA + \
                dI*np.exp(-f)*dA+I*(-df)*np.exp(-f)*dA+I*np.exp(-f)*d2A
        return result

#find M1,M2:
    def moment_1(self,s,texp,var_t):
        r_s = self.r(s)
        u_r = self.u(r_s,texp)
        u_mr = self.u(self.mr,texp)
        result =  self.dfai(u_r,texp,var_t) * self.du(r_s,texp) * self.dr(s)/ self.fai(u_mr,texp,var_t)
        return result

    def moment_2(self,s,texp,var_t):
        r_s = self.r(s)
        u_r = self.u(r_s,texp)
        u_mr = self.u(self.mr,texp)
        result =(self.d2fai(u_r,texp,var_t) * (self.du(r_s,texp)**2 )* (self.dr(s)**2)+   \
                 self.dfai(u_r,texp,var_t) * self.d2u(r_s,texp) * (self.dr(s)**2) +      \
                 self.dfai(u_r,texp,var_t) * self.du(r_s,texp) * self.d2r(s))       \
                 / self.fai(u_mr,texp,var_t)
        return result


    #find value
    def cond_spot_sigma(self, texp):
        var_t = self.var_t(texp)
        rhoc = np.sqrt(1.0 - self.rho ** 2)
        m1 = - self.moment_1(0,texp,var_t)
        m2 = self.moment_2(0,texp,var_t)
                 
        if self.dist == 0:
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            mu = m1
            lam = m1 ** 3 / (m2 - m1 ** 2)
            int_var_std = self.rng.wald(mean=mu, scale=lam) / texp
        elif self.dist == 1:
            scale_ln = np.sqrt(np.log(m2) - 2 * np.log(m1))
            miu_ln = np.log(m1) - 0.5 * scale_ln ** 2
            int_var_std = self.rng.lognormal(mean=miu_ln, sigma=scale_ln) / texp
        else:
            raise ValueError(f"Incorrect distribution.")

        ### Common Part
        int_var_dw = ((var_t - self.sigma) - self.mr * texp * (self.theta - int_var_std)) / self.vov
        spot_cond = np.exp(self.rho * (int_var_dw - 0.5 * self.rho * int_var_std * texp))
        sigma_cond = rhoc * np.sqrt(int_var_std / self.sigma)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond

print('end')
        
