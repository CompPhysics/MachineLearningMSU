import numpy as np
import matplotlib.pyplot as plt

def boot(data, statistic, R):
    from numpy import zeros, arange, mean, std, loadtxt
    from numpy.random import randint
    from time import time
    
    t = zeros(R); n = len(data); inds = arange(n); t0 = time()
    
    # non-parametric bootstrap
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])
    return t


def tsboot(data,statistic,R,l):
    from numpy import std, mean, concatenate, arange, loadtxt, zeros, ceil
    from numpy.random import randint
    from time import time
    t = zeros(R); n = len(data); k = int(ceil(float(n)/l));
    inds = arange(n); t0 = time()
    
    # time series bootstrap
    for i in range(R):
        # construct bootstrap sample from
        # k chunks of data. The chunksize is l
        _data = concatenate([data[j:j+l] for j in randint(0,n-l,k)])[0:n];
        t[i] = statistic(_data)
    return t

def block(x):
    from numpy import log2, zeros, mean, var, sum, loadtxt, arange, \
                  array, cumsum, dot, transpose, diagonal, floor
    from numpy.linalg import inv
    # preliminaries
    d = log2(len(x))
    if (d - floor(d) != 0):
        print("Warning: Data size = %g, is not a power of 2." % floor(2**d))
        print("Truncating data to %g." % 2**floor(d) )
        x = x[:2**int(floor(d))]
    d = int(floor(d))
    n = 2**d
    s, gamma = zeros(d), zeros(d)
    mu = mean(x)

    # estimate the auto-covariance and variances 
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])
   
    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q = array([3.841459,  5.991465,  7.814728,  9.487729,  11.070498, 12.591587, 
               14.067140, 15.507313, 16.918978, 18.307038, 19.675138, 21.026070, 
               22.362032, 23.684791, 24.995790, 26.296228, 27.587112, 28.869299, 
               30.143527, 31.410433, 32.670573, 33.924438, 35.172462, 36.415029, 
               37.652484, 38.885139, 40.113272, 41.337138, 42.556968, 43.772972])

    # use magic to determine when we should have stopped blocking
    for k in arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
    return s[k]/2**(d-k), s/2**(d - arange(0,d))


# autocorr. estimator
def acorr(X):
    Z = np.fft.ifft(abs(np.fft.fft(X - np.mean(X),2*len(X)))**2).real
    return Z[:len(X)]/Z[0]

# user input
mu, sigma,d = 0., 1., 16;
print("Use this script to generate a random time series with given mean, standard deviation and size")
print("For the generated time series, a selection of statistics are known exactly.")
print("After the time series has been generated, a choice of estimates for these quantities are presented.")
print("")
print("Select a mean, and standard deviation for the time series. For best readability select mean = 0, std.dev. = 1")
mu = float(input("Input some mean. This can be any real number. Then hit the Enter-key. "))
sigma = float(input("Input some standard deviation. This has to be a positive number. "))
sigma = abs(sigma)
d = int(input("Pick some integer d such that 10 < d < 20. Otherwise your computer will suffer. The size of the time series will equal 2**d. "))

# gen. residuals
n = 2**d; tau = n/10000.; phi = np.exp(-1/tau);
e = np.random.normal(0, sigma*(1 - phi**2)**.5, n)


# gen. AR(1) series
X = np.zeros(n); X[0] = mu + e[0]
for i in range(1,n):
    X[i] = phi*X[i-1] + e[i] + mu*(1 - phi)

plt.subplot(311); plt.plot(X); plt.title("Time series(y-axis) versus time (x-axis)"); plt.subplot(312);
plt.hist(X,50, None, True); plt.legend(["Relative frequency"]); plt.subplot(313)
h = np.arange(n)
plt.plot(h, phi**h, h, acorr(X)); plt.xlabel(r"$h$"); plt.xlim(0,2**4*tau)
plt.legend([r"Exact acf $\rho(h)/\sigma^2$",r"Acf ML estimate, $\widehat{\rho}(h)/\sigma^2$"]); plt.show()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

t = boot(X, np.mean, 1000)
fig, ax = plt.subplots(2, 1)
ax[0].text(0,.8,r"Two estimators of $\frac{\sigma}{\sqrt{n}}_{\mathrm{exact}} = \frac{%g}{%g} = %g$, is presented below." % (sigma, n**.5, sigma/n**.5))
ax[0].text(0,.6,r"Estimate 1: A maximum likelihood estimator. $\frac{\widehat{\sigma}}{\sqrt{n}} = %g$" % (np.std(X)/n**.5) ) 
ax[0].text(0,.4, r"Estimate 2: A bootstrap estimator. $\widehat{\sqrt{ \mathrm{Var}(\overline{X}) }}_{\mathrm{boot}} = %g$. See bootstrap distr. below" % (np.std(t)) )
ax[0].axis("off")
ax[1].hist(t, 50, None, True)
ax[1].set_title(r"Bootstr. distr. $\overline{X}_{\mathrm{boot}}$ and $\overline{X}_{\mathrm{boot}}$ + $\widehat{\sqrt{ \mathrm{Var}(\overline{X}) }}_{\mathrm{boot}}$. is indicated (estimated bias = %g)." % (np.mean(t) - mu ), fontsize=11 )
ax[1].set_xticks([np.mean(t), np.mean(t) + np.std(t)])
plt.show()



t = tsboot(X, np.mean, 1000, 3*tau)
h = np.arange(1,n)
fig, ax = plt.subplots(3, 1)
ax[0].text(0,.8,r"Three estimators of $\mathrm{Var}(\overline{X})_{\mathrm{exact}} = \frac{\sigma^2}{n} + \frac{2}{n} \sum_{h=1}^{n-1} \Big(1 - \frac{h}{n} \Big)\gamma(h) = %g$, is presented below." % (sigma**2/float(n) + 2.*sigma**2/n*np.sum( (1 - h/n)*phi**h) ))
ax[0].text(0,.6,r"Estimate 1: $\frac{\widehat{\sigma}^2}{n} + \frac{2}{n} \sum_{h=1}^{3\tau} \Big(1 - \frac{h}{n} \Big)\widehat{\gamma}(h) = %g$, $\tau$ time coefficient of $\gamma$" % (np.var(X)/float(n) + 2.*sigma**2/n*np.sum( (1 - h[1:3*tau]/n)*acorr(X)[1:3*tau]) ) ) 
ax[0].text(0,.4, r"Estimate 2: A time series bootstrap estimator. $\widehat{\mathrm{Var}(\overline{X}) }_{\mathrm{tsboot}} = %g$. See bootstrap distr. below" % (np.var(t)) )
ax[0].text(0,.2, r"Estimate 3: A blocking-method estimate. $\widehat{\mathrm{Var}(\overline{X}) }_{\mathrm{block}} = %g$. See illustration at the bottom" % ( block(X)[0] ))
ax[0].axis("off")
ax[1].hist(t, 50, None, True)
ax[1].set_title(r"Bootstr. distr. $\overline{X}_{\mathrm{tsboot}}$ and $\overline{X}_{\mathrm{tsboot}}$ + $\sqrt{\widehat{ \mathrm{Var}(\overline{X}) }_{\mathrm{tsboot}}}$. is indicated (estimated bias = %g)." % (np.mean(t) - mu ), fontsize=11 )
ax[1].set_xticks([np.mean(t), np.mean(t) + np.std(t)])
ax[2].plot(block(X)[1]); ax[2].set_xlabel("Blocking iterations"); ax[2].set_ylabel("Blocking estimate")
plt.show()
