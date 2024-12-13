# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:06:35 2024

@author: n.patsalidis
"""

class Bezier(MathAssist):
    def __init__(self,xvals, params,  M = None ):
        self.xvals = xvals
        self.params = params
        self.L = params[0]
        self.y0 = params[1]
        self.ye = params[-1]
        # 1 y defines bezier y points.y[0] = y[1] = y[ 2 ] := ycontrol[1], y[-3]=y[-2] =y[-1] := ycontrol[-1]
        ### L := y[0] and x[i] is equidistant from 0 to L
        y = np.empty(params.shape[0]+3,dtype=np.float64)
        y[2:-2] = params[1:]
        y[0] = self.y0
        y[1] = self.y0
        y[-1] = self.ye
        y[-2] = self.ye
        self.ycontrol = y
        self.npoints = y.shape[0]
        dx = self.L/float(self.npoints-1)
        x = np.empty_like(y)
        for i in range(self.npoints):
            x[i] = float(i)*dx
        self.xcontrol = x
        
        if M is None:
            self.M = self.matrix(self.npoints)
        else:
            self.M = M
        return
    
    #@staticmethod
    def matrix_coef(self,i,j,N):
        s = (-1)**(j-i)
        nj = self.numba_combinations(N, j)
        ij = self.numba_combinations(j,i)
        mij = s*nj*ij
        return mij
    
    #@staticmethod
    def matrix(self,Npoints):
        N = Npoints - 1 # Npoints = (N+1 sum is from 0 to N)
        M = np.zeros((Npoints,Npoints))
        for i in range(Npoints):
            for j in range(i,Npoints):      
                M[i,j] = self.matrix_coef(i,j,N)
        return M
   
    #@staticmethod 
    def u_vectorized(self):
        # 1  find taus using newton raphson from x positions(rhos)
        taus = self.find_taus__vectorized()
        # 1 end
        self.taus = taus
        # 1 find u using y control points (implicitly from coeff_y_tj)
        u = self.rtaus__vectorized(taus)
        # 1 end
        return u 
    #@staticmethod
    def find_coeffs_i(self,taus,M):
        n = M.shape[0]
        C = np.zeros((taus.shape[0], n))
        taus_power = np.ones((taus.size,))
        for i in range(n):
            taus_power = np.ones((taus.size,))
            for j in range(n):
                C[:,i] += M[i,j]*taus_power
                taus_power *= taus
        return C
            
    #@staticmethod 
    #@jit
    def rtaus__vectorized(self,taus, ):
        ny = self.npoints
        y = self.ycontrol
        M = self.M
        coeff_y_tj = np.zeros((ny,))
        for i in range(ny):
            ry = y[i]
            for j in range(i,ny):
                mij = M[i,j]
                coeff_y_tj[j] += ry * mij
        yr = np.zeros((taus.size,))  # Initialize yr with the same shape as taus
        taus_power = np.ones((taus.size,))
    
        for j in range(ny):
            yr += coeff_y_tj[j] * taus_power
            taus_power *= taus  
        return yr
    
    #@staticmethod 
    def find_taus__vectorized(self,):
        nx = self.npoints
        x = self.xcontrol
        M = self.M
        coeff_x_tj = np.zeros((nx,))
        
        for i in range(nx):
            rx = x[i]
            for j in range(i,nx):
                mij = M[i,j]
                coeff_x_tj[j] += rx * mij
        
        xvals = self.xvals
        taus = np.empty_like(xvals)
        #tguess = np.zeros((xvals.size,))
        tguess = xvals/self.L
        #return tguess
        fup = tguess>=1.0
        f0 = tguess == 0
        taus[fup] = 1.0
        taus[f0] = 0.0
        nf = np.logical_not(np.logical_or(f0,fup))
        taus[nf] = self.find_t__newton__vectorized(xvals[nf], coeff_x_tj, tguess[nf])
        return taus
    
    #@staticmethod 
    def find_t__newton__vectorized(self,xvals,coeff_tj,tguess):
        tol = 1e-8
        res = np.ones((xvals.size,))
        told = tguess
        tnew = tguess
        correctionFunc = self.rtdrdt_ratio__vectorized
        while np.any(res > tol):
            
            tnew = told - correctionFunc(told,coeff_tj,xvals)
            
            res = np.abs(tnew-told)
            told = tnew
    
        return tnew
    
    #@staticmethod 
    def rtdrdt_ratio__vectorized(self,taus,coeff_tj,xvals):
        n = coeff_tj.size
        nt = (taus.size,)
        dx = np.zeros(nt)
        x = np.zeros(nt)
        tj = x + 1.0
        tm1 = 1/taus
        for j in range(n):
            cmj = coeff_tj[j]
            dx+=float(j)*cmj*tj*tm1
            x +=cmj*tj
            tj *=taus
        return (x-xvals)/dx