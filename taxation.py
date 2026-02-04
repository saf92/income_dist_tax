from packages import *

def find_closest_index(vector, value):
    '''
    Find index of vector with value closest to inputed value.
    '''
    vector = np.array(vector)  # Ensure it's a NumPy array
    index = np.argmin(np.abs(vector - value))
    return index

def inc_after_band_tax(s,b,t):
    '''
    Inputs: s: income, b: income band points, t: tax bands
    Outputs: income after band tax
    '''
    s = np.sort(s)
    n = len(b) # len(t) = n+1
    for i in range(n):
        if i == 0:
            s_at = s[s<b[i]]*(1-t[i])
        elif (i>0) & (i<n-1):
            a = b[0]*(1-t[0])
            if i>1:
                for j in range(1,i):
                    a += (b[j]-b[j-1])*(1-t[j])
            k = a+(s[(s>=b[i-1])&(s<b[i])]-b[i-1])*(1-t[i])
            s_at = np.concatenate((s_at,k))
        else:
            a = b[0]*(1-t[0])
            for j in range(1,i):
                a += (b[j]-b[j-1])*(1-t[j])
            k = a+(s[(s>=b[i-1])&(s<b[i])]-b[i-1])*(1-t[i])
            s_at = np.concatenate((s_at,k))
            a = b[0]*(1-t[0])
            for j in range(1,i+1):
                a += (b[j]-b[j-1])*(1-t[j])
            k = a+(s[s>=b[i]]-b[i])*(1-t[i+1])
            s_at = np.concatenate((s_at,k))
        
    return s_at

def inc_after_tax_func(x,a1,a2):
    '''
    Simple power law function that fits well to intervals of income after tax.
    '''
    return a1*(x)**a2

def income_bands(s,b):
    '''
    Inputs: sample incomes s, income bands b
    Output: indices of income bands
    '''
    inds = []
    n = len(b)
    ind = np.where(s<b[0])[0]
    inds.append(ind)
    for i in range(n-1):
        ind = np.where((s>=b[i])&(s<b[i+1]))[0]
        inds.append(ind)
    ind = np.where(s>=b[n-1])[0]
    inds.append(ind)
    return inds

def income_tax_shares(s,s_at,b):
    '''
    Inputs: s: income sample, s_at: income after tax of sample and b: income band cutoffs.
    Output: tax shares by income band, if s_at = 0 (or any integer) tax bands are 0 and 
    outputs income shares before tax.
    '''
    sb_inds = income_bands(s,b)
    inc_s = []
    if type(s_at) != type(s):
        for i in range(len(sb_inds)):
            inc_sv = np.sum(s[sb_inds[i]])/np.sum(s)
            inc_s.append(inc_sv)
    else:
        for i in range(len(sb_inds)):
            inc_sv = (np.sum(s[sb_inds[i]])-np.sum(s_at[sb_inds[i]]))/(np.sum(s)-np.sum(s_at))
            inc_s.append(inc_sv)
    return inc_s

def gradients_tax_share(s,b):
    '''
    Find gradients/proportions of tax share function.
    '''
    a1, a2, a3 = b[0], b[1], b[2]
    N = np.sum(s)
    I = income_bands(s,b)
    s_b = income_tax_shares(s,0,b)
    n1, n2, n3 = len(I[1]), len(I[2]), len(I[3])
    S1, S2, S3 = s_b[1], s_b[2], s_b[3]
    g1 = S1 - n1*a1/N + (n2+n3)*(a2-a1)/N
    g2 = S2 - n2*a2/N + n3*(a3-a2)/N
    g3 = S3 - n3*a3/N
    return g1, g2, g3

def top_x_share(s,x):
    '''
    Inputs: s: income sample, x: x*100 top income share
    Returns: x*100 top income share
    '''
    n = len(s)
    x_i = n-int(n*x)
    S = np.sum(s)
    s = np.sort(s)
    return np.sum(s[x_i:])/S

def prop_tax(s_at,s):
    '''
    Inputs: s_at: sample of income after tax, s: sample of income before tax
    Outputs: share of total tax to total income
    '''
    return 1-np.sum(s_at)/np.sum(s)

def prop_tax1(s,b,t):
    '''
    Inputs: s: income sample, s_at: income after tax of sample and b: income band cutoffs.
    Outputs: share of total tax to total income.
    '''
    s_at = inc_after_band_tax(s,b,t)
    return prop_tax(s_at,s)

def prop_tax2(s,b,t):
    '''
    Should give same output as prop_tax1 using the gradients/proportions.
    '''
    g = gradients_tax_share(s,b)
    t = np.asarray(t,float)
    return np.sum(g*t[1:])

def income_after_tax(s):
    '''
    UK after tax function on single income s.
    '''
    if s <= 10**5:
        d = 12570
    elif 10**5 < s < 125140:
        d = 12570-(s-10**5)/2
    else:
        d = 0

    
    if s <= 12570:
        s_at = s
    elif s-d <= 37700:
        s_at = (s-d)*(1-0.2-0.08) + d
    elif 37700 < s-d <= 125140:
        s_at = (s-d-37700)*(1-0.4-0.02) + (37700)*(1-0.2-0.08) + d
    else:
        s_at = (s-125140)*(1-0.45-0.02) + (125140-37700)*(1-0.4-0.02) + 37700*(1-0.2-0.08)

    return s_at

def income_after_tax1(s_v):
    '''
    UK after tax function on vector of incomes s_v.
    '''
    s_ats = []
    for s in s_v:
        s_at = income_after_tax(s)
        s_ats.append(s_at)
    s_ats = np.asarray(s_ats,float)
    return s_ats

def income_after_tax_func1(x,g,p):
    '''
    Income after tax function.
    '''
    return p*x+(1-p)/g*(1-np.exp(-g*x))

def income_after_tax_func2(x,a,g,p,m):
    '''
    Negative income after tax function.
    '''
    return p*x+(a-p)/g*(1-np.exp(-g*x))+m

def marginal(x):
    '''
    Marginal function (approximation of first derivative) of after tax function.
    '''
    return x[1:]-x[:-1]