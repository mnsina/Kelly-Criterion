# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:45:35 2024

@author: HP
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import math

from intersect import intersection

# Parameters

r1 = 0.10
r2 = 0.15

c11 = 0.2**2
c22 = 0.3**2

rho = -0.5
c12 = math.sqrt(c11)*math.sqrt(c22)*rho

r = pd.DataFrame([r1,r2])
r_t = np.transpose(r)


c = pd.DataFrame([[c11,c12], [c12,c22]])
c_i = np.linalg.inv(c)


# Portfolio

w = np.linspace(0, 1, 1000)

rp = r1*w+r2*(1-w)
cp = c11*w*w + c22*(1-w)*(1-w)+2*c12*w*(1-w)
cp = np.sqrt(cp)

# Minimum Variance

w_minVar = (c22 - c12) / (c11 + c22 - 2 * c12)

r_mv = r1*w_minVar+r2*(1-w_minVar)
c_mv = math.sqrt(c11*w_minVar*w_minVar + c22*(1-w_minVar)*(1-w_minVar)+2*c12*w_minVar*(1-w_minVar))


# Kelly Return

w_optimal = (r1 - r2 + c22 - c12) / (c11 + c22 - 2 * c12)

r_k = r1*w_optimal +r2*(1-w_optimal )
c_k = math.sqrt(c11*w_optimal*w_optimal  + c22*(1-w_optimal )*(1-w_optimal )+2*c12*w_optimal *(1-w_optimal))




# Optimización Min Var (Validación)

fun = lambda x: (c11*x[0]**2 + c22*x[1]**2 + 2*c12*x[0]*x[1])

cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})

bnds = ((0, None), (0, None))

res = minimize(fun, (1, 0), method='SLSQP', bounds=bnds,
               constraints=cons)

print(res.fun)
print(res.x)


# Optimización Kelly (Validación)

fun = lambda x: ( -r1*x[0] - r2*x[1] + 0.5*( c11*x[0]**2 + c22*x[1]**2 + 2*c12*x[0]*x[1]) )

cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})

bnds = ((0, None), (0, None))

res = minimize(fun, (1, 0), method='SLSQP', bounds=bnds,
               constraints=cons)

print(res.fun)
print(res.x)


# Optimización Activo Libre de Riesgo (3 Activos)

r3 = 0.08
c33 = 0.0

C = pd.DataFrame([[c11, c12, 0], [c12, c22, 0], [0, 0, 0]])
R = [r1, r2, r3]
L = pd.DataFrame({'X1':[0.0], 'X2': [0.0], 'X3': [0.0]})
Results = pd.DataFrame({'Returns':[0.0], 'Std Dev':[0.0]})

for t in range(1, 601):

 Aux = np.linspace(r3, max(r1,r2)*1.33, 600)   
    
 fun = lambda x: (x@C@x)

 cons = ( {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1}, 
        {'type': 'eq', 'fun': lambda x: x@R -Aux[t-1]} )

 bnds = ((None, None), (None, None), (None, None))

 res = minimize(fun, (1, 0, 0), method='SLSQP', bounds=bnds,
               constraints=cons)

 L.loc[t-1] = np.round(res.x, 5)
 Aux2 = pd.DataFrame({'Returns':np.round([Aux[t-1]], 5), 'Std Dev':np.round([res.fun**(1/2)], 5)})
 Results.loc[t-1] = Aux2.iloc[0]
print(res.fun)
print(res.x)

x, y = intersection(cp, rp, Results['Std Dev'], Results['Returns'])


# Optimización Kelly Multiple

fun = lambda x: (-x@R + 1/2*x@C@x)

cons = ( {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1})

bnds = ((None, None), (None, None), (None, None))

res = minimize(fun, (1, 0, 0), method='SLSQP', bounds=bnds,
               constraints=cons)

r_k2 = R@res.x
c_k2 = (res.x@C@res.x)**(1/2)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(cp, rp, label="Efficient Frontier", color="blue")
plt.scatter(c_k, r_k, label="Kelly Portfolio", color="red")
plt.scatter(c_mv, r_mv,  label="MV Portfolio", color="green")
plt.plot(Results['Std Dev'], Results['Returns'], label="Capital Market Line", color="yellow")
plt.scatter(x[0], y[0],  label="Tanget Portfolio", color="pink", marker='x')
plt.scatter(c_k2, r_k2,  label="Kelly Portfolio (Leverage)", color="orange", marker='x')
plt.annotate('100% Asset #1', xy = (c11**(1/2), r1))
plt.annotate('100% Asset #2', xy = (c22**(1/2), r2))
plt.annotate('100% Asset #3', xy = (c33, r3))

plt.title("Modern Portfolio Theory | 2 Assets Model")
plt.xlabel("Portfolio Std Dev (\u03C3)")
plt.ylabel("Portfolio Return ( \u03BC)")
plt.legend()
plt.grid(True)
plt.show()


# Comparativa Función kelly

#r_k2-1/2*c_k2**(2)
#r_k-1/2*c_k**(2)
#y[0]-1/2*x[0]**(2)