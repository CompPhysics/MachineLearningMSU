import numpy as np
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

def f(x):
    return np.sin(x) +0.5*x

def MakePlot(x,y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label = labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)

x = np.linspace(-2*np.pi, 2*np.pi, 50)
fit = np.polyfit(x, f(x), deg=9)
#fity = np.polyval(fit, x)


matrix = np.zeros((3+1,len(x)))
matrix[3,:] = np.sin(x)
matrix[2,:] = x**2
matrix[1,:] = x
matrix[0,:] = 1

fit = np.linalg.lstsq(matrix.T, f(x), rcond =None)[0]
fity = np.dot(fit,matrix)
print(np.mean( (f(x)-fity)**2))

MakePlot([x,x], [f(x), fity], ['b', 'r.'], ['f(x)', 'Regfit'], ['x','f(x)'])
plt.show()
