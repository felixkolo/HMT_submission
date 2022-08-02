# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:31:25 2020

@author: Felix
"""

import os
import random
import math
import shutil

trainingDir = os.path.join(os.getcwd(), '..', 'Sim', 'Training')
testDir = os.path.join(trainingDir, '..', 'Test')
imageTrainingDir = os.path.join(trainingDir, 'image')
imageTestDir = os.path.join(testDir, 'image')
with open(os.path.join(trainingDir, 'resultsAll.dat'), 'r') as f:
    trainSet = f.readlines()

x=0.09090909090909090909
# testSetLength = (len(text)*x)/(len(text)*(1-x)) # results in 10% of data in test set
testSetLength= math.ceil(0.1*len(trainSet)*(1-x))

# testSet = []
# for i in range(testSetLength):
#     a = int(random.uniform(0, len(trainSet)))
#     testSet.append(trainSet[a])
#     trainSet.pop(a)

# with open(os.path.join(trainingDir, 'resultsAll.dat'), 'w') as f:
#     for line in trainSet:
#         f.write(line)

# with open(os.path.join(testDir, 'resultsAll.dat'), 'w') as f:
#     for line in testSet:
#         f.write(line)

# for i, line in enumerate(testSet):
#     linesplit = line.split(',')
#     index = int(linesplit[0])
#     filename = 'image_'+str(index)+'.png'
#     shutil.move(os.path.join(imageTrainingDir, filename), os.path.join(imageTestDir,filename))

result = []
for line in open(os.path.join(testDir, 'resultsAll.dat')).readlines():
    line = line.strip()
    linesplit = line.split(',')
    index = int(linesplit[0])

    thisLine = {"index":index, "line":line}
    result.append(thisLine)
sortedList = sorted(result, key=lambda k: k["index"])

# with open(os.path.join(testDir, 'resultsAll.dat'), 'w') as f:
#     for i,line in enumerate(sortedList):
#         f.write(line["line"]+'\n')