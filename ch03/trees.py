'''
Created on Mar 11, 2018
Decision Tree Source Code for Machine Learning in Action Ch03
@author: shanks10
'''

from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    # 特征名
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 计算dataSet的熵
def calcShannonEnt(dataSet):
    # 数据集数据个数
    # dataSet的格式如下，每个实例的最后一个元素表示这个实例的类别
    # [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    numEntries = len(dataSet)
    labelCounts = {}
    # dataSet中各个类别的个数，比如'yes'类有2个，'no'类有3个
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    # dataSet的熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 对dataSet的每个实例，返回符合第axis个特征取值为value的所有实例
# dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# axis = 0, value = 1,a1 = splitDataSet(dataSet, axis, value)
# 则a1 = [[1, 'yes'], [1, 'yes'], [0, 'no']],
# axis = 0, value = 0,a0 = splitDataSet(dataSet, axis, value)
# 则a0 = [[1, 'no'], [1, 'no']]
# 即用第axis个特征划分dataSet,axis取值为1的划分到a1中，axis取值为0的划分到a0中
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # dataSet中的实例的第axis个特征取值为value,则删掉这个实例的第axis元素（删掉这个特征,用这个特征划分数据集，
        # 划分后得到的各个子集自然不包括这个特征），然后加入retDataSet
        if featVec[axis] == value:
            # 以下两句将featVec中的第axis元素删掉，然后赋给reducedFeatVec
            # 相当于tmp = featVec
            # del(tmp[axis])
            # retDataSet.append(tmp)
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 第i个特征划分dataSet的信息增益最大
# dataSet格式要求，dataSet是由列表元素组成的列表，且列表元素有相同的数据长度
# dataSet的每个实例（列表元素）的最后一个元素是当前实例的类别
def chooseBestFeatureToSplit(dataSet):
    # 数据集共有numFeatures个特征
    numFeatures = len(dataSet[0]) - 1
    # 划分数据集前的熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 对每个特征计算信息增益，得到信息增益最大的特征的索引
    for i in range(numFeatures):
        # 第i个特征的所有取值（或者可以理解成实例第i个属性的取值）
        featList = [example[i] for example in dataSet]
        # 去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 第i个特征的取值为value
        for value in uniqueVals:
            # 用第i个特征取值为value划分dataSet得到的子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 用第i个特征划分dataSet后的熵（即划分后的每个子集的熵的权重和）
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 第i个特征的信息增益
        infoGain = baseEntropy - newEntropy
        # 找到numFeatures个特征中信息增益最大的特征，返回该特征索引
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 返回classList中出现次数最多的元素，本节中返回该子集中出现次数最多的类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items,
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet, labels):
    # dataSet中的所有实例的类别
    classList = [example[-1] for example in dataSet]
    # dataSet中的所有实例的类别相同，则dataSet返回实例类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # dataSet中已无可划分的特征，但是dataSet的实例类别并不统一，
    # 这种情况返回dataSet中出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 最好的划分特征的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    # 最好划分特征的所有取值
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 这个特征划分dataSet后得到各个子集，对每个子集创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# inputTree是之前构造好的决策树，featLabels是特征名集合，testVec是待分类实例
def classify(inputTree,featLabels,testVec):
    # 第一个划分的特征名
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 第一个划分的特征的索引
    featIndex = featLabels.index(firstStr)
    # testVec中第featIndex个特征的取值
    key = testVec[featIndex]
    # 决策树中根据firstStr划分后分支为key的子集
    # 若子集是字典，说明还没到叶子节点，继续向下找，直到找到叶子节点
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# 将创建好的决策树存到filename文件中
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

#将filename文件中的决策树读取出来
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)