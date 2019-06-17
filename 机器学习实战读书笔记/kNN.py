'''
k-近邻算法

优点：
    精度高、对异常值不敏感、无数据输入假定。
缺点：
    计算复杂度高、空间复杂度高。
适用数据范围：
    数值型、标签型。
'''
import os
import numpy as np
import heapq as hp
def createDataset():   #数据集的建立 包括数据的数值以及标签。
    group = np.array([[1.0,1.1],[1.0,1.0],[0.1,0.1],[0,0.1]])
    labels = ['A','A','B','B']

    return group,labels



def classify0(inX,dataSet,labels,k):
    dataSize=dataSet.shape[0]
    diffmat = np.tile(inX,(dataSize,1))-dataSet
    sqDiffmat = diffmat**2
    sqDistances = sqDiffmat.sum(axis=1)
    myheap = []
    for i in range(dataSize):
        hp.heappush(myheap,(sqDistances[i],labels[i]))
    labelDict={}
    ct=1
    while(myheap):
        if(ct>k):
            break
        x=hp.heappop(myheap)
        if(x[1] in labelDict):
            labelDict[x[1]]=labelDict[x[1]]+1
        else:
            labelDict[x[1]]=1
        ct+=1
    res=sorted(labelDict.items(),key=lambda x:x[1],reverse=True)
    #print(res)
    return res[0][0]

def autoNorm(dataSet):  #数据归一化
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    dataSize = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(dataSize,1))
    normDataSet = dataSet / np.tile(ranges,(dataSize,1))
    return normDataSet

def datingClasstest(testSet,testLabels,dataSet,labels,k): #错误率测试
    m=testSet.shape[0]
    ct=0.0
    for i in range(m):
        lbl=classify0(testSet[i],dataSet,labels,k)
        if(lbl != testLabels[i]):
            ct+=1
    return ct/float(m)


'''
手写识别系统
1，将图像格式转换为分类器使用的向量格式
2，将测试集数据进行测试
'''
def img2vector(filename):
    fileobj = open(filename)
    vector = np.zeros((1,1024))
    for i in range(32):
        linestr = fileobj.readline()
        for j in range(32):
            vector[0,j+i*32]=linestr[j]
    return vector

def handWritingClassTest():
    hwLabels=[]
    trainingPath = "C:\\Users\\AndyLee\\ML-workspace-python\\机器学习实战\\trainingDigits"
    trainingFlieList = os.listdir(trainingPath)
    length=len(trainingFlieList)
    dataSet = np.zeros((length,1024))
    for i in range(length):
        fileName=trainingFlieList[i]
        fileStr=fileName.split('.')[0]
        classNum= int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        filePath = trainingPath+"\\"+fileName
        dataSet[i,:] = img2vector(filePath)
        
    testPath = "C:\\Users\\AndyLee\\ML-workspace-python\\机器学习实战\\testDigits"
    testFlieList = os.listdir(testPath)
    length=len(testFlieList)
    ct =0.0
    for i in range(length):
        fileName = testFlieList[i]
        fileStr = fileName.split('.')[0]
        num = int(fileStr.split('_')[0])
        filePath = trainingPath+"\\"+fileName
        tmpvec=img2vector(filePath)
        
        res=classify0(tmpvec,dataSet,hwLabels,2)
        if(num!=res):
            ct+=1
        print("the classifier came back with: %d, the real answer is : %d"%(res,num))

    print("the total error is : %d"%ct)
    print("the total error  rate is : %f"%(ct/length))



handWritingClassTest()

#path = "C:\\Users\\AndyLee\\ML-workspace-python\\机器学习实战\\trainingDigits"

    
