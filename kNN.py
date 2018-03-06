'''
Created on Mar 6, 2018
kNN: k Neareest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (MxN)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison(should be an odd number)

Output:     the most popular class label

@author: shanks10
'''
from numpy import *
import operator
from os import listdir

# dataSet中距离inX最近的k个数据中，类别个数最多的类别就是inX的类别
def classify0(inX, dataSet, labels, k):
    # 训练数据集的行数
    dataSetSize = dataSet.shape[0]
    # 待求数据inX减去训练数据集的每个数据
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    # inX与训练集每个数据的距离
    distances = sqDistances**0.5
    # 距离排序后的索引
    # sortedDistIndicies[0] = distances最小值的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 距离最近的k个数据的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # k个数据中类别为voteIlabel的个数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对classCount.items()按第二个元素（类别个数）排序，返回classCount.items的排序结果
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 提取filename文件中的数据，
# returnMat是dataSet
# classLabelVector是labels
def file2matrix(filename):
    fr = open(filename)
    # 文件的行数
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    with open(filename) as fr:
        for line in fr:
            line = line.strip()
            # 每行数据分隔符是'\t'，返回list
            listFromLine = line.split('\t')
            # 每行数据前3个是特征，最后一个是类别，可根据实际情况修改
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
    return returnMat, classLabelVector

# 对dataSet (MxN)中的数据进行归一化处理
def autoNorm(dataSet):
    # 获得dataSet每一列的最小值，minVals (1xN)
    minVals = dataSet.min(0)
    # 获得dataSet每一列的最大值，maxVals (1xN)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # dataSet的行数
    m = dataSet.shape[0]
    # dataSet的每个元素都减去所在列的最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 元素级的除法，不是矩阵除法
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 将32x32的图像数据提取到1x1024的numpy数组中(每幅图像都要转成1x1024的numpy数组)
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        # 读取一行
        lineStr = fr.readline()
        for j in range(32):
            returnVect = [0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 以list存储trainingDigits文件夹中的文件名
    # trainingDigits文件夹中的文件命名格式为'0_10.txt'
    # 其中0_10表示数字0的第10幅图像
    trainingFileList = listdir('trainingDigits')
    # m个训练样本
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 文件名'0_10.txt'
        fileNameStr = trainingFileList[i]
        # 0_10
        fileStr = fileNameStr.split('.')[0]
        # 类别0
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 'trainingDigits/0_10.txt' 这幅图像转成1x1024的numpy数组
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    # 测试数据集
    testFileList = listdir('testDigits')
    # 分类错误个数
    errorCount = 0.0
    # mTest个测试样本
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("\n the total number of errors is： %d" % errorCount)
    print("\n the total error rate is: %f" % (errorCount / float(mTest)))
