from levin.common.utils import Cheb_nodes
from levin.hermite import levin_hermite_integral
import numpy as np
from levin.bernstein import levin_bernstein_integral

a = -1
b = 1
N = 5
s = np.full(N,1,dtype=int)
s[0] = 2
s[-1] = 2
tau = Cheb_nodes(a,b,N)
w_range = np.arange(10,110,10)

def f(x):
	# return np.sin(x)
	# if(x==1):
	# 	return 0
	# else:
	# return x*(1-x)/np.sqrt(1+x**2)
	return (1/(1+25*x**2)-1/26)
	# return x*(1-x)

def g(x):
	# return -(x-0.1)**2
	return x

ints = levin_hermite_integral(f,g,tau,s,w_range)
# ints = levin_bernstein_integral(f,g,tau,w_range)

print(ints)

# pl.figure()
# pl.plot(w_range,ints.real)
# pl.show()
