from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math as m
from numpy import linalg as LA

#data preprocessing
f = open('data_train_PNN.txt','r')
bacabaris = f.readlines()
x1 = []
x2 = []
x3 = []
y = []
for i in range(1,len(bacabaris)):
    splitted = bacabaris[i].replace('\n','').replace('\r','').split('\t')
    x1.append(float(splitted[0]))
    x2.append(float(splitted[1]))
    x3.append(float(splitted[2]))
    y.append(float(splitted[3]))

f = open('data_test_PNN.txt','r')
bacabaris = f.readlines()
xtest = []
kelastest = [[],[],[],[]]
for i in range(1,len(bacabaris)):
    splitted = bacabaris[i].replace('\n','').replace('\r','').split('\t')
    xtest.append([float(splitted[0]),float(splitted[1]),float(splitted[2])])
    kelastest[0].append(float(splitted[0]))
    kelastest[1].append(float(splitted[1]))
    kelastest[2].append(float(splitted[2]))
    
kelas0 = [[],[],[],[]]
kelas1 = [[],[],[],[]]
kelas2 = [[],[],[],[]]
xkelas0 = []
xkelas1 = []
xkelas2 = []

for i in range(len(y)):
    if (y[i] == 0):
        kelas0[0].append(x1[i])
        kelas0[1].append(x2[i])
        kelas0[2].append(x3[i])
        kelas0[3].append(y[i])
        xkelas0.append([x1[i],x2[i],x3[i]])
    elif (y[i] == 1):
        kelas1[0].append(x1[i])
        kelas1[1].append(x2[i])
        kelas1[2].append(x3[i])
        kelas1[3].append(y[i])
        xkelas1.append([x1[i],x2[i],x3[i]])
    elif (y[i] == 2):
        kelas2[0].append(x1[i])
        kelas2[1].append(x2[i])
        kelas2[2].append(x3[i])
        kelas2[3].append(y[i])
        xkelas2.append([x1[i],x2[i],x3[i]])

#data plotting
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kelas0[0], kelas0[1], kelas0[2], c='b', marker='o')
ax.scatter(kelas1[0], kelas1[1], kelas1[2], c='y', marker='v')
ax.scatter(kelas2[0], kelas2[1], kelas2[2], c='g', marker='s')
ax.scatter(kelastest[0], kelastest[1], kelastest[2], c='r', marker='x')
ax.set_xlabel('att1')
ax.set_ylabel('att2')
ax.set_zlabel('att3')
plt.grid()
plt.show()
