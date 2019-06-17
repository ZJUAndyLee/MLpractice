'''
决策树：
优点：计算复杂度不高，输出结构易于理解，对中间值的缺失不敏感，可以处理不相关特征数据。
缺点：可能会产生过度匹配的问题。
'''

'''
决策树创建的伪码描述：
createBranch():
    if(检测数据集中的每个子项是否都属于同一类)：
        return 类标签
    else：
        寻找划分数据集最好的特征
        根据特征划分数据集
        创建分支节点
        for x in 每个分支结点：
            递归调用createBranch()
        return 产生的所有分支结点

'''

'''
信息增益与熵：
'''
import math
def createDataSet():
    dataSet=[[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0;
        labelCounts[currentLabel]+=1

    shannonEnt=0.0
    for key in labelCounts:
        tmpd = float(labelCounts[key])/numEntries
        shannonEnt+= - tmpd*math.log2(tmpd)
    
    return shannonEnt

#分离数据
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if(featVec[axis]==value):
            tmpVec=featVec[:axis] #横向截取，但是不包括axis
            tmpVec.extend(featVec[axis+1:]) #纵向截取包括axis+1
            retDataSet.append(tmpVec)
            
    return retDataSet

#计算信息增益从而选择最好的数据划分方式
def chooseBestFeatureTosplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#把每一维特征提取出来
        valSet = set(featList)#将featList转化为一个set
        newEntroy = 0.0
        for value in valSet:
            subDataset = splitDataSet(dataSet,i,value)
            prob = float(len(subDataset))/len(dataSet)
            newEntroy+=prob*calcShannonEnt(subDataset)#这里是条件熵所以必须是当前分类的不同期望下的熵的加权
        infoGain=baseEntropy-newEntroy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#多数表决解决无法确定分类的问题
def majoritycnt(classList):
    classCount = {}
    for vote in classList:
        if vote in classCount.keys():
            classCount[vote]+=1
        else:
            classCount[vote]=1
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]
   
#创建决策树
def creatTree(dataSet,labels):
    classList = [exm[-1] for exm in dataSet]
    if(classList.count(classList[0])==len(classList)):
        return classList[0]
    if(len(dataSet[0])==1):
        return majoritycnt(classList)
    bestFeat = chooseBestFeatureTosplit(dataSet) #得到最好的分类特征的维度
    bestFeatLabel = labels[bestFeat] #得到最好分类特征
    myTree = {bestFeatLabel:{}} #初始化子树
    del(labels[bestFeat])   #得到范围更下的标签
    featValues = [exm[bestFeat] for exm in dataSet]
    uniqueValues = set(featValues)  #得到该分类下的不同的取值
    for value in uniqueValues:
        subLabels=labels[:] 
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value),subLabels) #根据分类的维度和不同的取值把数据分开 再递归调用从而建成树
    return myTree
        
def getNumLeafs(myTree):
    numLeaf=0
    firstStr = list(myTree.keys())[0]
    subDict = myTree[firstStr]
    for key in list(subDict.keys()):
        if(type(subDict[key]).__name__=='dict'):
            numLeaf += getNumLeafs(subDict[key]) 
        else:
            numLeaf+=1
    return numLeaf

def getTreeDepth(myTree):
    maxDepth=0
    firstStr = list(myTree.keys())[0]
    subDict = myTree[firstStr]
    for key in list(subDict.keys()):
        if(type(subDict[key]).__name__=='dict'):
            thisDepth=1+getTreeDepth(subDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth

'''
决策树图的绘制
'''
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeaf=0
    firstStr = list(myTree.keys())[0]
    subDict = myTree[firstStr]
    for key in list(subDict.keys()):
        if(type(subDict[key]).__name__=='dict'):
            numLeaf += getNumLeafs(subDict[key]) 
        else:
            numLeaf+=1
    return numLeaf


def getTreeDepth(myTree):
    maxDepth=0
    firstStr = list(myTree.keys())[0]
    subDict = myTree[firstStr]
    for key in list(subDict.keys()):
        if(type(subDict[key]).__name__=='dict'):
            thisDepth=1+getTreeDepth(subDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

#分类器
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    subDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)#找到当前类属于第几维
    for key in list(subDict.keys()):#遍历当前分类中所有的关键词即所有的种类 找到与目标相匹配的集合
        if(testVec[featIndex]==key):
            if(type(subDict[key]).__name__=='dict'):#判断是否为子节点 是的话直接返回标签否则递归调用classify
                classType=classify(subDict[key],featLabels,testVec)
            else:
                classType=subDict[key]
    return classType

#在磁盘上储存以及得到的决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

fr = open('lenses.txt')
lenses=[tmp.strip().split('\t') for tmp in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
testLabels=lensesLabels[:]
lensesTree=creatTree(lenses,testLabels)
createPlot(lensesTree)
