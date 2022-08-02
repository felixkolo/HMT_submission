# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:38:32 2020

@author: Felix
"""
from scipy.spatial import Voronoi
# from scipy.spatial import voronoi_plot_2d
# import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import os
from abq_setup import *

pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize = stp()

def createVoronoi(pth, b):
    start_time = time.time()

    dirName = str(b + 1)
    if not os.path.exists(pth + '/Sim/SimFolders/' + dirName) and b + 1 < 65535:
        os.makedirs(pth + '/Sim/SimFolders/' + dirName)
        print("Directory ", dirName,  " Created ")
#        break
    else:
        print("Directory ", dirName,  " already exists")
#        next
    width = W * 1.5
    height = H * 1.5
    x = []
    y = []
    t_consume = []
    # parameter of the first random point
    x.append(random.uniform(0.01 * width, width - 0.01 * width))
    y.append(random.uniform(0.01 * height, height - 0.01 * height))
    j = 0
    max_vor_pts = int(np.random.uniform(30, 180))
    # max_vor_pts = 180
    dis_min = (1/18000)*(max_vor_pts-180)**2+0.75 # linear: 30pts -> distance = 2
                                    # 180 pts -> distance = 0.5
    tic = time.process_time()
    while True:
        x_neu = random.uniform(0.01 * width, width - 0.01 * width)
        y_neu = random.uniform(0.01 * height, height - 0.01 * height)
        j = 0
        for i in range(0, len(x)):
            #        print(i)
            # check whether the minimum distance to other points is kept:
            if (math.sqrt(pow(x_neu - x[i], 2) + pow(y_neu - y[i], 2)) > dis_min):
                j = j + 1
            else:
                break
        if j == len(x):
            x.append(x_neu)
            y.append(y_neu)
            toc = time.process_time()
            t = toc - tic
            t_consume.append(t)
        if j == max_vor_pts:  # maximum seed points for voronoi diagram reached
            break
    # print('creating %d points takes %.20f
    # seconds'%(len(x),t_consume[len(t_consume)-1]))
    # points
    points = np.array([])
    for i in range(len(x)):
        points = np.append(points, [x[i], y[i]], axis=None)
    points = points.reshape(int(len(points) / 2), 2)
    vor = Voronoi(points)

    # plot
    # voronoi_plot_2d(vor, line_width=2, line_alpha=1, point_size=5)
    # axes = plt.gca()
    # # axes.axis('equal')
    # axes.set_xlim([2.5,12.5])
    # axes.set_ylim([2.5,12.5])


    vor_vertices = vor.vertices
    vor_ridges = vor.ridge_vertices

    # write vertices into text file
    f = open(pth + '/Sim/SimFolders/' + dirName +
              '/' + 'voronoi_vertices.txt', 'w')
    f.write('** Vertices\n')
    for row in vor_vertices:
        np.savetxt(f, row)
    f.close()
    # write ridges into text file
    f = open(pth + '/Sim/SimFolders/' + dirName +
              '/' + 'voronoi_ridges.txt', 'w')
    f.write('** Ridges\n')
    for row in vor_ridges:
        np.savetxt(f, row)
    f.close()
    # write voronoi seeds into text file
    f = open(pth + '/Sim/SimFolders/' + dirName +
              '/' + 'num_voronoi_seeds.txt', 'w')
    f.write('** Number of voronoi seeds\n'+str(max_vor_pts))
    f.close()

#    # write setup variables into text file
#    f=open(pth+'/Sim/'+dirName+'/'+'setup.txt', 'w')
#    f.write(str(width)+' ')
#    f.write(str(height)+' ')
#    f.close()


    # plt.figure()
    # # Mark the Voronoi vertices.
    # plt.plot(vor.vertices[:,0], vor.vertices[:, 1], 'ko', ms=8)
    # plt.ylabel('width coordinate')
    # plt.xlabel('height coordinate')

    # for vpair in vor.ridge_vertices:
    #     if vpair[0] >= 0 and vpair[1] >= 0:
    #         v0 = vor.vertices[vpair[0]]
    #         v1 = vor.vertices[vpair[1]]
    #         # Draw a line from v0 to v1.
    #         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)
    # #        plt.show()

    elapsed_time = time.time() - start_time
    # takes around 0.1 seconds for each folder
    print('shit takes %.3f seconds ' % elapsed_time)

if __name__ == '__main__':
    createVoronoi(pth, b)

