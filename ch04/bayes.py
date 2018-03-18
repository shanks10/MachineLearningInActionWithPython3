'''
Created on Mar 14, 2018

@author: shanks10
'''


from numpy import *


def loadDataSet():
    postingList =[['my','dog','has','flea',\
                  'problems','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',\
                 'i','love','him'],
                  ['stop','posting','stupid','worthless','garbage'],
                  ['mr','licks','ate','my','steak','how',\
                  'to','stop','him'],
                  ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])
    # dataSet 出现的所有单词放入vocabSet（去重）
    for document in dataSet:
        # vocabSet和set(document)的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 将inputSet转化为词向量，vocabList是词汇表
def setOfWords2Vec(vocabList, inputSet):
    # 构造和vaocbList相同大小的list，元素都为0
    # 如果词汇表的词出现在inputSet中，则returnVec
    # 对应位置的元素置为1
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    # 训练集数据个数（多少篇文档用来训练）
    numTrainDocs = len(trainMatrix)
    # 词汇表大小
    numWords = len(trainMatrix[0])
    # trainCategory元素为1的文档的概率
    # 即训练集中分类为1的文档的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 对每个类别，计算词汇表各个词汇出现的次数，
    # 且次数初始值为1，若为0，比如某个词w1没有出现过，
    # 则p(w1 | c) = 0,
    # 这样 p(w | c) = p(w1 | c) p(w2 | c) ... p(wn | c) = 0
    # 由此结果产生偏差
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 类别1中各个单词出现的次数
            p1Num += trainMatrix[i]
            # 类别1中单词总个数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1 各个单词的条件概率
    # 条件概率相乘时由于条件概率都很小，造成太多很小的数相乘
    # 存在下溢出问题，将ab转为log(ab)= log(a)+log(b)
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
    # 条件概率相乘转为取对数相加，pClass1为类别1的概率
    p1 = sum(vec2classify * p1Vec) + log(pClass1)
    p0 = sum(vec2classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classify as : ',classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as : ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    # 所有电子邮件
    docList = []
    classList = []
    fullText = []
    # email/spam/和email/ham/下各有25封电子邮件
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 随机选10封电子邮件做测试数据
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # 训练集
    trainMat = []
    # 训练集的样本对应的类别，1为spam，0为ham
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 用训练集求得条件概率和类别概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 测试集的分类结果和真实结果不一致，则分类错误
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print("the error rate is: ", float(errorCount / len(testSet)))

# fulltext出现频率最高的30个词
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    # 所有文档
    docList = []
    # 文档对应的类别
    classList = []
    # 所有文档，元素就是单词，不是list
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # fullText出现频率最高的30个词
    top30Words = calcMostFreq(vocabList, fullText)
    # 移除词汇表中出现频率最高的20个词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 训练集数据索引，比如docList有30个文档，则trainingSet就是list(range(0,30))
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(int(0.1 * len(trainingSet))):
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引
        testSet.append(trainingSet[randIndex])
        # 将添加到testSet的索引从trainingSet中删除，
        # 这样trainingSet就只包含训练集的索引
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 构造训练集
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 返回训练集得到的参数
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 测试集
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is : ", float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    # p0v,p1V中条件概率大于阈值的保存到topSF,topNY
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    # 返回该类别中的词汇以及对应的条件概率，并依条件概率排序，
    sortedSF = sorted(topSF, key=operator.itemgetter(1), reverse=True)
    sortedNY = sorted(topNY, key=operator.itemgetter(1), reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF")
    for item in sortedSF:
        print(item[0])
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY")
    for item in sortedNY:
        print(item[0])