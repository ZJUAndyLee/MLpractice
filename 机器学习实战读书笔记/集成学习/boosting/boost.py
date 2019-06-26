'''
boosting算法是基于单个学习器之间具有强依赖关系的算法
boosting的单个学习器之间是串联关系生成一个基学习器
单个学习器之间往往从前往后都是递进的关系，即前面的学习器无法解决的数据会在后面的学习器上有更大的权重
从而使得整个基学习器具有更强的学习能力
'''
import numpy as np

def loadSimpData():
    datMat=np.matrix([[1.0,1.0],[1.3,1.0],[1.0,2.1],[1.5,1.6],[2.0,1.0]])
    classLabels = [1.0,1.0,-1.0,-1.0,-1.0]
    return datMat,classLabels

#建立单个学习器，这里是单层决策树
def stumClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):#我们用一个单层的决策树作为我们的弱分类器
    dataMatrix=np.matrix(dataArr)
    labels=np.matrix(classLabels).T
    m,n=np.shape(dataMatrix)
    numStep=10.0;bestStump={};bestClasEst=np.mat(np.zeros((m,1)))#我们把分类的区间格点化然后每一步去判断误差最后再取区间误差的最小值
    minError=np.inf#将误差设为负无穷然后进行离散求解
    for i in range(n):#遍历数据的所有维度找到某一个最佳分类的维度
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max()
        #print(numStep)
        stepSize=(rangeMax-rangeMin)/numStep#离散化处理求出误差最小的分类
        for j in range(-1,int(numStep)+1):
            for inequal in ['lt','gt']:#用于标签不同所以对于一个二分数据我们需要判断大于小于两种情况
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.mat(np.ones((m,1)))
                errArr[predictedVals==labels]=0
                weightedError=np.matmul(D.T,errArr)
                #print("split: dim%d, thresh %.2f,thresh inequal: %s, the weighted error is %.3f"%(i,threshVal,inequal,weightedError))
                if(weightedError<minError):
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst#最后我们到在当前误差权重下的最小分类的维度，数值，大小于

def adaBoostTrainDs(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)#初始化我们的数据权重，最开始设为1/m
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):#我们需要多少个基分类器进行多少次循环
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)#得到当前权重小的误差
        #print("D.T: ",D.T)
        alpha=float(0.5*np.log((1.0-error)/max(error,1e-16)))#得到分类器的权重alpha
        bestStump['alpha']=alpha
        #print("alpha:",alpha)
        weakClassArr.append(bestStump)
        #print("classEst: ",classEst.T)
        expon=np.multiply(-1*alpha*np.mat(classEst),np.mat(classLabels).T)#如果预测正确那么误差权重下降，如果预测错误那么对应的误差权重升高
        D=np.multiply(D,np.exp(expon))#根据分类的权重对下一次进行迭代的数据权重D进行更新
        D/D.sum()
        aggClassEst+=alpha*classEst
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("total Rate: ",errorRate)
        if(errorRate==0.0):
            break
    return weakClassArr

#基于Adaboost算法的分类
def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
    return np.sign(aggClassEst)

def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fileobj=open(filename)
    for line in fileobj.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(len(curLine)-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

dataArr,labels=loadDataSet('horseColicTraining2.txt')
testArr,testLabels=loadDataSet('horseColicTest2.txt')
classifierArr=adaBoostTrainDs(dataArr,labels,500)
predArr=adaClassify(testArr,classifierArr)
m=len(predArr)
errorArr=np.mat(np.zeros((m,1)))
errorArr[np.mat(testLabels).T!=predArr]=1.0

print(errorArr.sum()/m)

