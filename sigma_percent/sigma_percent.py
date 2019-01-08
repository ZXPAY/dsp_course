# -*- coding: utf-8 -*-
# calculate normal distribution, different sigma, ? percent


import numpy as np

mu = 0
sigma = 1
dataNumbers = 50001

X = np.linspace(-10, 10, dataNumbers, dtype=np.float64)
p_x = 1/(((2*np.pi)**0.5)*sigma)*np.exp(-((X-mu)**2)/(2*sigma**2))
proportion = np.sum(p_x)   # all area is 1

# test fifteen sigmas
sigma_index = np.linspace(1, 16, 16)/2
for i in range(sigma_index.shape[0]):
    index_l = np.where(X==-sigma*sigma_index[i])[0][0]
    index_r = np.where(X==sigma*sigma_index[i])[0][0]
    
    
    area = np.sum(np.abs(p_x[index_l:index_r])) / proportion
    print(sigma_index[i], '\u03c3:', area)

