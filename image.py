# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:23:29 2020

@author: Felix
"""
import __main__
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import os.path
from abq_setup import stp


def image(pth, b):
    start_time = time.time()
    picDir = pth + "/Sim/Training/pic/"
    imgDir = pth + "/Sim/Training/image/"
    simDir = pth + "/Sim/SimFolders/" + str(b) + "/"
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
        print("Directory ", imgDir,  " Created ")

    # numFiles = len([name for name in os.listdir(picDir) if os.path.isfile(os.path.join(picDir, name))])
    # print(numFiles)
    # for b in range(1,numFiles+1):
    if not os.path.isfile(imgDir + 'image_' + str(b) + '.png'):
        # import and read image into RGBA-array
        img = mpimg.imread(picDir + 'materialprobe' + str(b) + '.png')
        #imgplot = plt.imshow(img)

        # lists for saving counters were to cut the image
        k = []
        l = []

        # check if Pixel is "empty" until half of image height (for excluding coordinate system from image)
        while True:
            for i in range(len(img)):
                for j in range(int(len(img[0]) / 2)):
                    if round(img[i, j, 0].item(), 5) == 0.99608:
                        next
                    else:
                        k.append(i)
                        l.append(j)
                        break
            break

        # cut image: top:bottom, left:right
        img = img[k[0]:k[0] + int(len(k)),
                  l[0]:l[0] + int(len(l)), :]

        # # replace black pixels with blue pixels
        # for i in range(len(img)):
        #     for j in range(int(len(img[0]))):
        #         A = round(img[i, j, 0], 5)
        #         A = A.item()
        #         A = round(A, 5)
        #         B = 0.0
        #         if A == B:
        #             # new RGBA values for each black pixel
        #             img[i, j, 0] = 7 / 255
        #             img[i, j, 1] = 132 / 255
        #             img[i, j, 2] = 216 / 255
        #             img[i, j, 3] = 1
        #             next

        mpimg.imsave(imgDir + 'image_' + str(b) + '.png', img,
                     vmin=None, vmax=None, cmap=None, format=None)
        # mpimg.imsave(simDir + 'image_' + str(b) + '.png', img,
        #              vmin=None, vmax=None, cmap=None, format=None)
        # os.remove(picDir + 'materialprobe' + str(b) + '.png')
    elapsed_time = time.time() - start_time
    print(elapsed_time)  # takes around 12 seconds


if __name__ == '__main__':
    pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize = stp()
    image(pth, b)
