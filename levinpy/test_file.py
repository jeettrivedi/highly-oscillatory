from Hermite_Levin import Levin_Int
import matplotlib.pyplot as pl
import numpy as np
from helper_methods import Cheb_nodes
import sympy as sp
from Levin_Bernstein import Levin_Bern_int

a = -1
b = 1
N = 5
s = np.full(N,1,dtype=int)
s[0] = 2
s[-1] = 2
tau = Cheb_nodes(a,b,N)
w_range = np.arange(10,110,10)

def f(x):
	# return sp.sin(x)
	# if(x==1):
	# 	return 0
	# else:
	# return x*(1-x)/sp.sqrt(1+x**2)
	return (1/(1+25*x**2)-1/26)
	# return x*(1-x)

def g(x):
	# return -(x-0.1)**2
	return x

ints = Levin_Int(f,g,tau,s,w_range)
# ints = Levin_Bern_int(f,g,tau,w_range)

print(ints)

pl.figure()
pl.plot(w_range,ints.real)
# pl.show()
