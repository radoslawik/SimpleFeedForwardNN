# -*- coding: utf-8 -*-
"""
Simple implementation of Feedforward Neural Network (FFNN)
"""

import numpy as np
import matplotlib.pyplot as plt

N = 2 #number of variables
K = 10 #number of neurons in the layer

x1, x2, y = np.loadtxt("input.data", delimiter =' ', usecols =(0, 1, 2),  unpack = True)

I = len(x1) # amount of input data
J = 3 # three possibke outputs

x1 = np.asarray(x1).reshape(-1,1)
x2 = np.asarray(x2).reshape(-1,1)
y = np.asarray(y).reshape(-1,1)

for i in range(0,len(y)):
    clr = ''
    if(y[i]==0):
        clr = 'ro'
    elif(y[i]==1):
        clr = 'bo'
    else:
        clr = 'go'
    plt.plot(x1[i], x2[i], clr)
plt.axis([0, 10, 0, 10])
plt.title("Actual output variable")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.show()

x0 = np.ones((I,1)) 
XI = np.concatenate((x0, x1 ,x2), axis=1) # extended X matrix

# first layer neuron matrix
V = np.array([[-0.50131844,  0.65838527, -0.60157993,  0.65302492,  0.294982  ,
        -0.84132428, -0.04401377, -0.48535996, -0.48050531,  0.12579815],
       [ 0.51153216,  0.53274982,  0.39745388, -0.97512427, -0.39862787,
        -0.17849767,  0.12963851,  0.98837412, -0.56181012, -0.69005636],
       [-0.6424869 , -0.92725727,  0.85884549, -0.1419165 ,  0.02142941,
        -0.41971407,  0.37333187, -0.20431422,  0.86330821,  0.3062777 ]])

# second layer neuron matrix
W = np.array([[ 0.98940933, -0.34778822,  0.18073294],
       [-0.30859373, -0.75752076,  0.10272493],
       [-0.5390539 ,  0.83161405,  0.00342125],
       [-0.36501278, -0.53333388, -0.29562208],
       [ 0.34750455, -0.45474235,  0.70110718],
       [-0.67197557, -0.48580859, -0.44963163],
       [-0.88702443, -0.10803092, -0.55720584],
       [-0.71052302, -0.16704237, -0.08458558],
       [-0.82211589, -0.07491992, -0.21067935],
       [ 0.22310839, -0.23244688,  0.22887257],
       [ 0.37120256,  0.03351009, -0.69234653]])


iteration = 0
SSE = []
alpha1= 0.1
alpha2= 0.01   
    
#------------------------------------------------------------------------------
#FORWARD PROPAGATION PART
for iteration in range(150): 
  
    XII = np.matmul(XI, V) 
    F = 1/(1+np.exp(-XII)) 
    FI = np.concatenate((x0,F),axis=1)     
    FII = np.matmul(FI, W)  
    G = 1/(1+np.exp(-FII))

    Y = np.zeros((I,J)) 
    for i in range(I):
        if (y[i]==0):
            Y[i][0]=1 ##[1,0,0]
        elif (y[i]==1):
            Y[i][1]=1 ##[0,1,0]
        elif (y[i]==2):
            Y[i][2]=1 ##[0,0,1]
                
    
    E = 0
    for i in range(I):
        for j in range(J):
            E+=0.5*(G[i][j] - Y[i][j])**2
    SSE.append(E)
    
    #------------------------------------------------------------------------------
    ##BACK PROPAGATION PART
  
    for k in range(K+1):
        for j in range(J):
            Sum = 0
            for i in range(I):
                Sum+=(G[i][j] - Y[i][j])*G[i][j]*(1-G[i][j])*FI[i][k]
            W[k][j] = W[k][j] - alpha1*Sum
    
    for n in range(N+1):
        for k in range(K):
            Sum = 0
            for i in range(I):
                for j in range(J):
                    Sum += (G[i][j] - Y[i][j])*G[i][j]*(1-G[i][j])*W[k][j]*FI[i][k]*(1-FI[i][k])*XI[i][n]
            V[n][k] = V[n][k]-alpha2*Sum



it = [i for i in range(1,len(SSE)+1)]
plt.plot(it,SSE)
plt.xlabel("epoch")
plt.ylabel("value")
plt.title("SSE")
plt.grid(True)
plt.show()

y_pred = []
for j in range(0,I):
    max_val = 0
    max_val = max(G[j][0],G[j][1],G[j][2])
    if G[j][0]==max_val:
        max_val=0
    elif G[j][1]==max_val:
        max_val=1
    else:
        max_val=2
    y_pred.append(max_val)
    
    
    
for i in range(0,len(y_pred)):
    clr = ''
    if(y_pred[i]==0):
        clr = 'ro'
    elif(y_pred[i]==1):
        clr = 'bo'
    else:
        clr = 'go'
    plt.plot(x1[i], x2[i], clr)
plt.axis([0, 10, 0, 10])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Predicted output variable")
plt.grid(True)
plt.show()
    

