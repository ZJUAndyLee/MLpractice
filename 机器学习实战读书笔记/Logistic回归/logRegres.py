'''
基于Logistic回归和Sigmoid函数的分类：
优点：
    计算代价不高，易于理解和实现。
缺点：
    容易欠拟合，分类精度不高。
sigmoid的函数一般用于模型的最后，直接来实现分类若是输入值大于0则分类为1，若是输入值小于0则分类为0

利用梯度上升来求解模型最大值。

算法伪码：
初始化参数值
for iter in iterator_times
    计算每个参数的梯度
    更新回归系数

'''

import numpy as np

def loadDataSet():
    dataMat=[]; labelMat=[]
    fileobj = open('testSet.txt','r')
    for line in fileobj.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha = 0.001 #学习率
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        tmp=sigmoid(np.matmul(dataMatrix,weights))
        error=labelMat-tmp
        #print(alpha*np.matmul(dataMatrix.transpose(),error))
        weights=weights+alpha*np.matmul(dataMatrix.transpose(),error)
    return weights

def stocGradAscent0(dataMatrix,classLabels):
    m,n=np.shape(dataMatrix)
    alpha=0.01
    weights=np.ones(n)
    for j in range(200):
        for i in range(m):
            h=sigmoid(sum(dataMatrix[i]*weights))
            error=classLabels[i]-h
            weights=weights+np.multiply(alpha*error,dataMatrix[i])
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter):
    import random
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        randIndex = [x for x in range(m)]
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            dataIndex = int(random.uniform(0,len(randIndex)))
            h=sigmoid(sum(dataMatrix[dataIndex]*weights))
            error=classLabels[dataIndex]-h
            weights=weights+np.multiply(alpha*error,dataMatrix[dataIndex])
            del(randIndex[dataIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if(int(labelMat[i])==1):
            xcord1.append(dataArr[i][1]);ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1]);ycord2.append(dataArr[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=np.array((-weights[0]-weights[1]*x)/weights[2])
    print(np.shape(x))
    ax.plot(x,y)
    plt.xlabel("X1");plt.ylabel("X2");
    plt.show()


'''
如何处理数据中的缺失值：
使用可用特征值的均值来补缺失值
使用特殊值来填补缺失值
忽略有缺失值的样本
使用相似样本的均值填补缺失值
使用另外的机器学习算法的预测值来填补缺失值
'''

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if(prob>0.5):
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        tmpLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(tmpLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(tmpLine[21]))
    trainingWeights=stocGradAscent1(np.array(trainingSet),trainingLabels,1000)
    errorcount=0.0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        tmpLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(tmpLine[i]))
        if(int(classifyVector(lineArr,trainingWeights))!=int(tmpLine[21])):
            errorcount+=1.0
    errorRate=errorcount/float(numTestVec)
    print("the error rate of this test is %f:"%errorRate)
    return errorRate

dataSet,labels=loadDataSet()
averageRate=0.0
for i in range(10):
    averageRate+=colicTest()
print("the total average Rate of %d times is %f: "%(10,float(averageRate)/10))
#print(stocGradAscent0(dataSet,labels))