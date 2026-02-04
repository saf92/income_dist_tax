from packages import *

def min_max(x):
    '''
    Min-max transformation of vector x.
    '''
    m, M = np.min(x), np.max(x)
    return (x-m)/(M-m)


def k_gen_exp(x,k):
    '''
    Inputs variable x and parameter k.
    Outputs the k-generalised exponential function.
    '''
    return (np.sqrt(1+k**2*x**2)+k*x)**(1/k)

def k_gen_log(x,k):
    '''
    Inputs variable x and parameter k.
    Outputs the k-generalised log function (inverse of exponential).
    '''
    return (x**k-x**-k)/(2*k)

def k_gen_tail(x,A,k,a,b):
    '''
    Inputs variable x and parameters A, k, a, b
    Outputs the tail of the (modified) k-generalised distribution
    '''
    x1= -b*x**a
    return A*k_gen_exp(x1,k)


def k_gen_fit(x,y,params0,g):
    '''
    Inputs: x values of tail distribution, y values of tail distribution, 
    initial parameters params0 for fitting, and g value for weights
    Output: k-gen parameter fit
    '''
    bds = ([0,0,1,0], [2,1,2,10**-4])
    weights = y**g # sigma = 1/sqrt{weight}
    popt, pcov = curve_fit(k_gen_tail, xdata = x, ydata = y, p0 = params0, bounds = bds, sigma = weights)
    return popt

def param_fit(x,y,g):
    '''
    Inputs: x values of tail distribution, y values of tail distribution, 
    and g value for weights
    Output: errors and parameters of best fit after using many starting parameters
    '''
    ks = np.arange(0.5,1,0.1)
    a_s = np.arange(1.5,2.2,0.1)
    bs = [10**-8,10**-7,10**-6]
    errors = []
    params = []
    for k in ks:
        for a in a_s:
            for b in bs:
                params0 = [1.3,k,a,b]
                try:
                    popt = k_gen_fit(x,y,params0,g)
                except:
                    popt = params0
                params.append(popt)
                y_pred = k_gen_tail(x,*popt)
                error = np.sum((y-y_pred)**2)
                errors.append(error)
    ind = np.argmin(errors)
    error_min = errors[ind]
    popt = params[ind]
    return error_min, popt

def gen_k_gen_sample(n,A,k,a,b):
    '''
    Generate a sorted sample of size n from k-gen. distribution.
    '''
    p = uniform.rvs(size = n)
    s =(-(1/b)*k_gen_log(p/A,k))**(1/a)
    s = np.sort(s)
    return s

def get_x_m(A,k,a,b):
    '''
    Output the x_m for the modified k-gen. distribution.
    '''
    return (-1/b*k_gen_log(1/A,k))**(1/a)


def emp_tail(x):
    '''
    Empirical tail of data x 
    '''
    x = np.sort(x)
    n = len(x)
    y = np.arange(1/n,1+1/n,1/n)[::-1]
    return x,y


def gini(s):
    '''
    Input: sample of incomes s.
    Output: Gini coefficient of s.
    '''
    n = len(s)
    s = np.sort(s)
    i = np.arange(1,n+1)
    return 2/n*np.sum(i*s)/np.sum(s) - (n+1)/n

def Theil(s):
    '''
    Input: sample of incomes s.
    Output: Theil index of s.
    '''
    m = np.mean(s)
    return np.mean((s/m)*np.log(s/m))

def share_ratios(s,p):
    '''
    Inputs: sample of incomes s, quantiles p
    Output: income shares between quantiles
    '''
    S = np.sum(s)
    n = len(s)
    s = np.sort(s)
    n_p = len(p)
    shares = []
    for i in range(n_p):
        if i == 0:
            p_i = int(n*p[i])
            share = np.sum(s[:p_i])/S
            shares.append(share)
        else:
            p_i1, p_i2 = int(n*p[i-1]), int(n*p[i])
            share = (np.sum(s[:p_i2])-np.sum(s[:p_i1]))/S
            shares.append(share)
            if i == n_p-1:
                p_i = int(n*p[i])
                share = np.sum(s[p_i:])/S
                shares.append(share)
    return shares