import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

def e(x):
    return (x-1)*(x-2)*(x-3)*(x-5)

def De(x):
    return 4*x**3-33*x**2+82*x-61

def DG1(x0,mu, epsi=0.01, n_max=1000):
    x1 = x0-mu*De(x0)
    i = 0
    x = []
    y = []
    while i <= n_max:
        i = i + 1
        x1 = x0-mu*De(x0)
        if x1-x0 >= epsi :
            break
        x0 = x1
        x.append(e(x1))
        y.append(i)
    return x1, i, e(x1), x, y

"""
a,b,c,data, label = DG(5,0.001)

plt.figure()
plt.plot(data, label)
plt.show()
"""
"""
def derivePa(a,b,x,y):
    return 2*x*(a*x+b-y)
    
def derivePb(a,b,x,y):
    return 2*(a*x+b-y)

def degA(i, n ,a, b, x, y):
    return degA(i-1, n ,a, b, x, y) - n * derivePa(a,b,x,y) * degA(i-1, n ,a, b, x, y)
print(degA(5, 5,12,4,9,1))
"""

x, y = make_blobs(n_features=1, n_samples=100)

def dp_a(a, n, x):
    r = []
    for xi in x:
        r.append(2*xi[0])
    return a - (n * np.array(r).sum())

def dp_b(b, n, x):
    return b - (n * 2*len(x))
    
def E(a,b,x,y):
    r = []
    for xi, yi in zip(x,y):
        r.append(((a*xi[0]+b)-yi)**2)
    return np.array(r).sum()

def DG(a, b, x, y, n, nbMax):
    r = []
    r.append([a, b, E(a,b,x,y)])
    for k in range(nbMax):
        a = dp_a(a, n, x)
        b = dp_b(b, n, x)
        r.append([a, b, E(a,b,x,y)])
    return np.array(r)

print(DG(1, 1, x, y, 0.001, 100))
print(DG(1, 1, x, y, 0.001, 500))
print(DG(1, 1, x, y, 0.001, 1000))
print(DG(1, 1, x, y, 0.01, 1000))
print(DG(1, 1, x, y, 1, 1000))

