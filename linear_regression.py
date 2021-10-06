import pandas
import numpy
import matplotlib.pyplot as plt

path='D:\py\data.txt'
data=pandas.read_csv(path, header=None, names=['Population', 'Profit'])#获取数据

def Hypothesis(theta_0,theta_1,x):
    result=theta_0.T+numpy.multiply(theta_1.T,x)
    return result

def CostFunction(theta_0,theta_1,x, y):
    inner = numpy.power(((Hypothesis(theta_0,theta_1,x)) - y), 2)
    return numpy.sum(inner) / (2 * len(x))

def GradientDescent(x, y,theta_0,theta_1,alpha):
    global theta0,theta1
    j0=numpy.sum(Hypothesis(theta0,theta1,x)-y)/len(x)
    j1=numpy.sum(numpy.multiply((Hypothesis(theta0,theta1,x)-y),x))/len(x)
    result0=theta_0[0]-alpha*j0
    result1=theta_1[0]-alpha*j1
    theta0=numpy.full(len(x),result0)
    theta1=numpy.full(len(x),result1)
    

cols = data.shape[1]#返回data列数
x = data.iloc[:,:cols-1]#x是data里的除了最后一列以外所有列
y = data.iloc[:,cols-1:cols]#y是data最后一列
alpha = 0.0001#学习率
iters = 15000#迭代次数


#转换x和y为numpy矩阵
x = numpy.matrix(x.values)
y = numpy.matrix(y.values)
theta0 = numpy.full(len(x),0)
theta1 = numpy.full(len(x),0)

for i in range(iters):
    GradientDescent(x,y,theta0,theta1,alpha)



X = numpy.linspace(data.Population.min(), data.Population.max(), 100)
Y=theta0[0]+theta1[0]*X

fig,ax=plt.subplots(figsize=(9,6))
ax.plot(X, Y, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
