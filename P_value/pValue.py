# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:04:35 2018

@author: zxpay
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.special import gamma
from scipy import stats
from scipy.stats import f

mu = 0
sigma = 1
dataNumbers = 5000
PICKLE_FILE = 'brainCauseNumberDict.pickle'

normalD = np.random.normal(mu, sigma, dataNumbers)
X = np.linspace(-10, 10, dataNumbers)
p_x = 1/((2*np.pi*sigma**2)**0.5)*np.exp(-((X-mu)**2)/(2*sigma**2))


fig = plt.figure(figsize=(20,12))
#count, bins, ignored = plt.hist(normalD, 30, normed=True)
plt.plot(X, p_x)
plt.grid(True)
plt.title('Normal Distribution')
plt.show()

### Read pickle file about brain cause death
dataDict = {}
with open(PICKLE_FILE, 'rb') as f:
    dataDict = pickle.load(f)

Wdict = dataDict['woman']
Mdict = dataDict['man']

DIM = len(Wdict)
ageX = range(DIM)

Xticks = list(Wdict.keys())

Wdata = list(Wdict.values())
Mdata = list(Mdict.values())
WdataNumber = sum(Wdata)
MdataNumber = sum(Mdata)

Wdata = np.array(Wdata).astype(np.float32)
Mdata = np.array(Mdata).astype(np.float32)

### Visualize data
fig = plt.figure(figsize=(20,12))
plt.plot(ageX, Wdata, 'r')
plt.plot(ageX, Mdata, 'b')
plt.xticks(ageX, Xticks)
plt.grid(True)
plt.title('Brain cause death, Woman and Man', fontsize=40)
plt.xlabel('years', fontsize=30)
plt.ylabel('deaths', fontsize=30)
plt.legend(['Woman', 'Man'])
plt.savefig('picture.png')
plt.show()


WdeathMean = np.mean(Wdata)
WdeathStd = np.std(Wdata)
print('Woman mean is {:.2f} ,std is {:.2f}'.format(WdeathMean, WdeathStd))

MdeathMean = np.mean(Mdata)
MdeathStd = np.std(Mdata)
print('Man mean is {:.2f} ,std is {:.2f}'.format(MdeathMean, MdeathStd))

t = (MdeathMean - WdeathMean) / ((MdeathStd**2/MdataNumber + WdeathStd**2/WdataNumber)**0.5)

print('t value is ', t)


dof = WdataNumber + MdataNumber - 2
#f_dof = gamma(((dof+1)/2)) / (((dof*np.pi)**0.5)*gamma(dof/2)) * ((1+(X**2/dof))**(-(dof+1)/2))
### Error, because gamma(((dof+1)/2))  is inf, (((dof*np.pi)**0.5)*gamma(dof/2)) is inf

f_dof = max(p_x)*((1+(X**2/dof))**(-(dof+1)/2))

lineX = np.ones([dataNumbers])
liney = np.linspace(0, 0.4, dataNumbers)

fig = plt.figure(figsize=(20,12))
plt.plot(X, p_x, 'g<--')
plt.plot(X, f_dof, 'b>--')
plt.plot(lineX*t, liney, 'r', linewidth=2.)
plt.plot(lineX*-t, liney, 'r', linewidth=2.)
plt.grid(True)
plt.legend(['Normal', 't-distribution'])
plt.title('Normal & student-t Distribution', fontsize=40)
plt.xlabel('X', fontsize=30)
plt.ylabel('P(X)', fontsize=30)
plt.savefig('t_distribution.png')
plt.show()



### F-test
# step one:
F = MdeathStd**2/WdeathStd**2
print('F value is ', F)
M_dof = MdataNumber - 1
W_dof = WdataNumber - 1

alpha = 0.05
p_value = stats.f.cdf(F, M_dof, W_dof)
print('p value is {:.2f}'.format(p_value))
if p_value > alpha:
    print('Reject hypothesis that Var(X) == Var(Y)')
else:
    print('Accept !!!')



import scipy.stats as ss

def plot_f(x_range, dfn, dfd, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the f distribution function for a given x range, dfn and dfd
    If mu and sigma are not provided, standard f is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.f.cdf(x, dfn, dfd, mu, sigma)
    else:
        y = ss.f.pdf(x, dfn, dfd, mu, sigma)
    plt.plot(x, y, **kwargs)
    
    
x = np.linspace(0.001, 5, M_dof)
plot_f(x, M_dof, W_dof, 0, 1, color='red', lw=2, ls='-', alpha=0.5, label='pdf')
plt.legend();  


