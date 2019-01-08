# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
import pandas as pd
from numpy.linalg import inv
from sklearn.model_selection import train_test_split

df = pd.read_csv('A_LVR_LAND_A.csv', encoding='big5')
A = np.zeros([df.shape[0], df.shape[1]-1])
Y = np.zeros([df.shape[0], 1])

c = 0
for key in df.keys():
    if '總價元' == key:
        Y[:,0] = df[key]
        print('Y : ', key)
    else:
        A[:,c] = df[key]
        c+=1
        print('X-', c, ' : ', key)
Y = Y/1e6   # 除以百萬

### Generate data myself  ( Testing )
#A1 = np.linspace(1,100,1000).reshape(-1,1)
#A2 = np.linspace(50,100,1000).reshape(-1,1)
#A = np.hstack((A1, A2))
#Y = 3.14*A[:,0] - 1.65*A[:,1]
#Y = Y.reshape(-1, 1)
 
# Function Y = A * theta
# theta = (A.T * A)^-1 * A.T * Y

# 後面 20% 作測試
A_train, A_test, Y_train, Y_test = train_test_split(A, Y, test_size=0.2, shuffle=True)
theta = np.matmul((inv(np.matmul(A_train.T, A_train))), np.matmul(A_train.T, Y_train))
print('theta ', theta)

pred_Y_train = np.matmul(A_train, theta)
error_train = np.mean(np.abs(Y_train - pred_Y_train))
print('Error train : ', error_train)
dot_x = np.linspace(1, Y_train.shape[0], Y_train.shape[0])
plt.figure(figsize=(20,12))
plt.plot(dot_x, Y_train, 'b^--')
plt.plot(dot_x, pred_Y_train, 'r>--')
plt.xlabel('dot')
plt.ylabel('million($)')
plt.title('training (80%)')
plt.legend(['True', 'Predict'])
plt.show()

pred_Y_test = np.matmul(A_test, theta)
error_test = np.mean(np.abs(Y_test - pred_Y_test))
print('Error test : ', error_test)
dot_x = np.linspace(1, Y_test.shape[0], Y_test.shape[0])
plt.figure(figsize=(20,12))
plt.plot(dot_x, Y_test, 'b^--')
plt.plot(dot_x, pred_Y_test, 'r>--')
plt.xlabel('dot')
plt.ylabel('million($)')
plt.title('testing (20%)')
plt.legend(['True', 'Predict'])
plt.show()



