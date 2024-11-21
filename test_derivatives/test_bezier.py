
import sys
sys.path.append('../main_scripts')
import FF_Develop as ff
from numba.experimental import jitclass
from time import perf_counter
# Define the class specification (if you have attributes)
spec = []
import numpy as np
from matplotlib import pyplot as plt

ff.TestPotentials.demonstrate_bezier(seed=2)
#C,taus,M = ff.TestPotentials.test_coeffs_i()
L =3.12314
r = np.arange(0,L,0.01)
params = np.array([L,0,10,10,-7,-6,10,-5,6,10],dtype=float)
params[0] = L
t0 = perf_counter()
Nt=100
for _ in range(Nt):
    b = ff.Bezier(r,params) ; u = b.u_vectorized()
print("Time for u: {:4.3e} ms".format(1000*(perf_counter() -t0)/Nt))
t0 = perf_counter()
for _ in range(Nt):
    dy = b.find_dydyc_serial();  
print("Analytical (serial) time for  dydyc {:4.3e} ms".format(1000*(perf_counter() -t0)/Nt))
t0 = perf_counter()

for _ in range(Nt):
    dynumpy = b.find_dydyc_vectorized();  
print("Analytical (vectorized) time  for  dydyc {:4.3e} ms".format(1000*(perf_counter() -t0)/Nt))

t0 = perf_counter()
for _ in range(Nt):
    dyn = b.find_dydyc_numerically()
print("Numerical time for  dydyc {:4.3e} ms".format(1000*(perf_counter() -t0)/Nt))
diff = np.abs(dynumpy-dyn).max()
print('Differnce between control point derivatives:  max differnce = {:4.3e}'.format(diff)) 
b1 = ff.Bezier(r,params) ; u = b1.u_vectorized()
for _ in range(Nt):
    g = b1.find_params_gradient()
print("Analytical (serial) time for g {:4.3e} ms".format(1000*(perf_counter() -t0)/Nt))
b2 = ff.Bezier(r,params) ; #u = b2.u_vectorized()
for _ in range(1):
    g = b2.find_params_gradient()
print("Analytical (vectorized) time for g {:4.3e} ms".format(1000*(perf_counter() -t0)/Nt))
#g_num = np.empty_like(params)
def compute_high_precision_gradient( params, epsilon=1e-3):
    for i in range(params.shape[0]):
        # Create copies of params and perturb only the i-th parameter for 4th order central difference
        p2p = params.copy()
        p2p[i] += 2 * epsilon
        
        p1p = params.copy()
        p1p[i] += epsilon
        
        p1m = params.copy()
        p1m[i] -= epsilon
        
        p2m = params.copy()
        p2m[i] -= 2 * epsilon
        
        # Compute function values at these perturbed parameters
        b2p = ff.Bezier(r, p2p)
        u2p = b2p.u_vectorized()
        
        b1p = ff.Bezier(r, p1p)
        u1p = b1p.u_vectorized()
        
        b1m = ff.Bezier(r, p1m)
        u1m = b1m.u_vectorized()
        
        b2m = ff.Bezier(r, p2m)
        u2m = b2m.u_vectorized()
        
        # Fourth-order central difference formula
        g_num = (-u2p + 8 * u1p - 8 * u1m + u2m) / (12 * epsilon)
        g_num = (u1p-u1m)/(2*epsilon)
        # Debugging prints to track the perturbation and results
        #print(f"Parameter {i} perturbed by 2*epsilon, +epsilon, -epsilon, -2*epsilon")
        #print(f"u2p: {u2p}, u1p: {u1p}, u1m: {u1m}, u2m: {u2m}, g_num: {g_num}")
        if i ==0: g_num0=g_num
        # Compute the max difference between numerical and analytical gradients
        print("parameter {:d} --> max difference {:4.3e}".format(i, np.abs(g_num - g[i]).max()))
    return g_num0
    
g_num0 = compute_high_precision_gradient(params, epsilon=1e-3)

fig = plt.figure(figsize=(3.2,3.2),dpi=300)
plt.title('Gradient checking')
dr = r[1]-r[0]
plt.plot(r[1:-1],(u[2:]-u[:-2])*0.5/dr,label='dydx-num')
plt.plot(b2.xvals,b2.dydx,label='dydx-ana',ls='--')
plt.plot(b2.xvals,g_num0,label='g0-num',ls='-')
plt.plot(b2.xvals,b2.params_gradient[0],label='g0-a',ls='--')
plt.legend(frameon=False)
plt.show()

fig = plt.figure(figsize=(3.2,3.2),dpi=300)
plt.yscale('log')
plt.title('Error on dydL')
plt.ylim((1e-18,1))
plt.plot(b.taus,np.abs(g_num0-g[0]))
plt.show()

fig = plt.figure(figsize=(3.3,3.3),dpi=300)
n = dy.shape[0]
plt.title('Gradients over control points dydy<i>')
cmap = plt.get_cmap('viridis')
plt.xlabel('t')
for j in range(n):
    plt.plot(b.taus,dy[j],label=j,color=cmap(j/(n-1)))
plt.legend(frameon=False,ncol=2)
plt.show()
fig = plt.figure(figsize=(3.3,3.3),dpi=300)
plt.title('Linearity of x(t) if x are equally distant')
plt.ylabel('y(t),x(t)')
plt.plot(b.taus,b.xvals,label='x(t)')
plt.plot(b.taus,u,label='y(t)')
plt.legend(frameon=False,fontsize=7)
plt.show()

