'''
朴素贝叶斯：
优点：在数据较少的情况下仍然有效，可以处理多类别问题。
缺点：对于输入数据的准备方式较为敏感。
贝叶斯决策理论核心思想：
即：选择具有最高概率的决策。
'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocablist,inputVec):
    returnVec=[0]*len(vocablist)
    for word in inputVec:
        if(word in vocablist):
            returnVec[vocablist.index(word)]+=1;
    return returnVec

'''
文本分析，从词量分析概率，该问题我们最重要的是假设所有词的出现是互不干扰的。
所以对于长句或者文本出现的概率也就是为每一个词的概率之积
'''
'''
由于文本的出现次数可能为零所以单纯的直接相乘可能出现概率为零的现象而无法得到正真的结果
其次对于有文本出现词频基数很大的情况可能出现概率小于计算精度的情况 而我们取对数的话能够减缓这种趋势
'''

import numpy as np

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWord=len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWord)
    p1Num = np.ones(numWord)
    p0Total=2.0
    p1Total=2.0
    for i in range(numTrainDocs):
        if(trainCategory[i]==1):
            p1Num+=trainMatrix[i] #计算在1分类下每个词出现的词频
            p1Total+=sum(trainMatrix[i]) #计算1分类下所有词的词频之和
        else:
            p0Num+=trainMatrix[i]
            p0Total+=sum(trainMatrix[i])
    #print(p1Num)
    p0Vec=np.log(np.divide(p0Num,p0Total))
    p1Vec=np.log(np.divide(p1Num,p1Total))
    return p0Vec,p1Vec,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pclass1):
    p1= sum(np.multiply(vec2Classify,p1Vec))+np.log(pclass1)
    p0= sum(np.multiply(vec2Classify,p0Vec))+np.log(1.0-pclass1)
    if p1>p0:
        return 1
    else:
        return 0

def testNB():
    dataSet,classVec=loadDataSet()
    wordList=createVocabList(dataSet)
    finalSet=[]
    for inputVec in dataSet:
        finalSet.append(setOfWords2Vec(wordList,inputVec))
    p0Vec,p1Vec,pAbusive=trainNB0(finalSet,classVec)
    testEntry=['love','my','dalmation']
    thisDoc = setOfWords2Vec(wordList,testEntry)
    print(classifyNB(thisDoc,p0Vec,p1Vec,pAbusive))
    testEntry=['stupid','garbage']
    thisDoc = setOfWords2Vec(wordList,testEntry)
    print(classifyNB(thisDoc,p0Vec,p1Vec,pAbusive))

def textParse(bigString):
    import re
    tmpList=re.split(r'\W',bigString)
    return [word.lower() for word in tmpList if(len(word)>2)]

def spamTest():
    import random
    docList=[]; classList=[]; fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=[i for i in range(50)]; testSet=[]
    #找出十个文本向量来做我们的测试集数据
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        del(trainingSet[randIndex])
        testSet.append(randIndex)
    
    #剩下的40个文本向量作为训练集数据
    trainingMat = []; trainingClass=[]
    for i in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList,docList[i]))
        trainingClass.append(classList[i])
    p0V,p1V,pSpam=trainNB0(trainingMat,trainingClass)
    error=0.0
    for i in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[i])
        print(i)
        if(classifyNB(wordVector,p0V,p1V,pSpam)!=classList[i]):
            error+=1
    print('the error rate is: ',float(error)/len(testSet))
    


spamTest()