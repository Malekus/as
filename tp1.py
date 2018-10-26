import matplotlib.pyplot as plt

def e(x):
    return (x-1)*(x-2)*(x-3)*(x-5)

def De(x):
    return 4*x**3-33*x**2+82*x-61


def DG(x0,mu, epsi=0.01, n_max=1000):
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


a,b,c,data, label = DG(5,0.001)

plt.figure()
plt.plot(data, label)
plt.show()


def derivePa(a,b,x,y):
    return 2*x*(a*x+b-y)
    
def derivePb(a,b,x,y):
    return 2*(a*x+b-y)

def degA(i, n ,a, b, x, y):
    return degA(i-1, n ,a, b, x, y) - n * derivePa(a,b,x,y) * degA(i-1, n ,a, b, x, y)



print(degA(5, 5,12,4,9,1))