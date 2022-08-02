# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:55:21 2020

@author: Felix
"""
import os
# import time
import shutil
import re
pth = 'D:/Dokumente/Uni/Masterarbeit/Masterarbeit'  # Windows
pth_new_geom = os.path.join(pth, 'new-geometry', 'Sim')
pth_vf_var_geom = os.path.join(pth, 'vf-var-geometry', 'Sim')
resultsfile = os.path.join(pth, 'Sim', 'Training', 'resultsAll.dat')

for dirnum in range(41,47): # 1-10;41-50 new-geometry, 21-40 vf-var-geometry:
    # dirnum= 4
    directory = os.path.join(pth_new_geom, str(dirnum))
    destinyResultsDir = os.path.join(pth_new_geom, 'Training', 'resultsAll.dat')
    resultsDir = os.path.join(directory, 'Sim', 'Training', 'resultsAll.dat')
    resultsShifted = os.path.join(directory, 'Sim', 'Training', 'resultsShifted.dat')
    shutil.copyfile('' + resultsDir, '' + resultsShifted)
    count=0
    lastDir = 0
    with open(destinyResultsDir, 'r') as f:
        f = open(destinyResultsDir, 'r')
        for line in f:
            count+=1
            linesplit =line.split(',')
            lastDir = int(linesplit[0])
        f.close()
            # else:
                # lastDir = 0

    with open(resultsDir, 'r') as f:
        text = f.readlines()

    i=0
    with open(resultsShifted, 'w') as f:
        for line in text:
            linesplit =line.split(',')
            # if i != 0:
            line = str.replace(line, linesplit[0], str(int(linesplit[0])+lastDir), 1)
            f.write(line)
            i += 1

    with open(resultsShifted, 'r') as f:
        text = f.readlines()

    with open(os.path.join(pth_new_geom, 'Training', 'resultsAll.dat'), 'a') as f:
        for line in text:
            if line[0:3] != 'dir':
                f.write(line)

    # rename files from image dir 'image_xxx.png'
    imageDir = os.path.join(directory, 'Sim', 'Training', 'image')
    regex = re.compile(r'\d+')
    for filename in os.listdir(imageDir):
        old_num = [int(x) for x in regex.findall(filename)]
        old_num = old_num[0]
        new_num = old_num + lastDir
        shutil.copyfile(os.path.join(imageDir,'image_'+str(old_num)+'.png'), os.path.join(pth_new_geom, 'Training', 'image', 'image_'+str(new_num)+'.png'))

    # # rename files from pic dir 'materialprobexxx.png'
    # picDir = os.path.join(directory, 'Sim', 'Training', 'pic')
    # regex = re.compile(r'\d+')
    # for filename in os.listdir(picDir):
    #     old_num = [int(x) for x in regex.findall(filename)]
    #     if filename != 'withMesh':
    #         old_num = old_num[0]
    #         new_num = old_num + lastDir
    #         shutil.move(filename, 'materialprobe'+str(new_num)+'.png')

    # # rename files from withMesh dir 'materialprobexxx+mesh.png'
    # picmeshDir = os.path.join(directory, 'Sim', 'Training', 'pic', 'withMesh')
    # regex = re.compile(r'\d+')
    # for filename in os.listdir(picmeshDir):
    #     old_num = [int(x) for x in regex.findall(filename)]
    #     old_num = old_num[0]
    #     new_num = old_num + lastDir
    #     shutil.move(filename, 'materialprobe'+str(new_num)+'+mesh.png')