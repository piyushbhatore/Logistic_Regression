import matplotlib.pyplot as plt
import numpy as np
a = np.loadtxt('./data/data1.txt',delimiter=',')
a = np.array(a)
a[:,2] = a[:,2].astype(int)
columns = [i for i in range(a[:,0].size) if a[i,2]==1]
positive_data = a[columns,:]
ncolumns = [i for i in range(a[:,0].size) if a[i,2]==0]
negative_data = a[ncolumns,:]
plt.scatter(positive_data[:,0],positive_data[:,1])
plt.scatter(negative_data[:,0],negative_data[:,1],color='r')
# theata = [  0.13485425,
#    0.1205178 ,
#  -15.92889784]
theata = [  0.12768524,
   0.12189752,
 -15.32302349]

x = np.array(range(0,101)).astype(float)
y= (-1*theata[2] - theata[0]*x)/theata[1]
plt.scatter(x,y,color='g')
plt.show()