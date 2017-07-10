import numpy as np
def sig(x):						#defining the sigmoid function
	return 1 / (1 + np.exp(-x))
def cost(Y,Y_predict):
	#print(np.sum(np.multiply(Y,np.log(np.absolute(Y_predict)))+np.multiply((1-Y),np.log(np.absolute(1-Y_predict)))))
	return -1*np.sum(np.multiply(Y,np.log(np.absolute(Y_predict)))+np.multiply((1-Y),np.log(np.absolute(1-Y_predict))))/Y.size
sigmoid = np.vectorize(sig)
data = np.genfromtxt('data/data1.txt',delimiter=',',dtype=float)
data = np.array(data)
columns = data.shape[1]
rows = data.shape[0]
accuracy = 1000000



############      TRAINING         #################################################
train_data = data[0:int(0.75*rows),:]		## data size is 100
test_data = data[int(0.75*rows):rows,:]
Y = np.copy(train_data[:,columns-1])
Y = np.matrix(Y).transpose()
testY = np.matrix(np.copy(test_data[:,columns-1]))
train_data[:,columns-1] = np.ones([int(0.75*rows)])  ##adding bias
theta = np.ones((columns,1),dtype='f')
####data scaling
# avg = np.mean(train_data,axis=0);
# sd = np.std(train_data,axis=0);
# train_data[:,0:columns-1] = (train_data[:,0:columns-1] - avg[0:columns-1] )/ sd[0:columns-1];
##LOOP##
for k in range(accuracy):
	Y_predict = sigmoid(np.dot(train_data,theta));
	totalcost = np.dot(train_data.transpose(),Y_predict-Y) ## add regularizer term here
	theta = theta - 0.001*totalcost/train_data.shape[0];
	print(cost(Y,Y_predict))



#########      TEST        ###########################################################
test_data[:,columns-1] = np.ones([int(0.25*rows)])  ##adding bias
Y_predict = (Y_predict>0.5).astype(int)
print(np.multiply(Y,1-Y_predict)+np.multiply(1-Y,Y_predict))
print(theta)
print(cost(testY,sigmoid(np.dot(test_data,theta))))
print((sigmoid(np.dot(test_data,theta))>0.5).astype(int))