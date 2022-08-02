# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:08:45 2020

@author: Felix
"""
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import os
import numpy as np
import random
import math
import sys
from abq_setup import*
from load_voronoi import *
from abq_create_part import *
from abq_material import *
from abq_assembly import *
from abq_mesh import *
from abq_img import *
from abq_job import *
from abq_move_files import *
from abq_step import *
from abq_model2 import *
from postpro import *


def abqVor(pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize):
    # os.chdir(r"D:\Dokumente\Uni\Masterarbeit\Masterarbeit\Python")
    # pth = "D:\Dokumente\Uni\Masterarbeit\Masterarbeit"
    session.journalOptions.setValues(
        replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
    # print(pth)
    os.chdir(r"" + os.path.join(pth, 'Work'))
    sys.path.append(os.path.join(pth, 'Python'))
    # Setup
    # pth,b,W,H,d_min,d_max,strut_width,Qp,meshSize = stp()

    # Load voronoi vertices and ridges
    vor_vertices, vor_ridges, num_vor_pts = loadVor(pth, b)

    # Abaqus
    # part
    i, j = createPart(pth, W, H, vor_vertices, vor_ridges,
                      d_min, d_max, strut_width, num_vor_pts)

    # Material properties
    abqMat(i)

    # assembly
    Vf = abqAss(i, j)

    # step creation, defining boundary conditions and heat flux load
    edges1, edges2 = abqStep(W, Qp, d_min)

    # mesh
    numnodesL, numnodesR = abqMesh(meshSize, W, edges1, edges2)

    # picture
    abqImg(pth)

    # job creation, writing and execution
    abqJob(pth)

    # second simulation
    edges3, edges4, numnodesB, numnodesT = abqmodel2(pth, Qp, H)

    # move files
    moveFiles(pth, b)

    # # # # # call postpro script (possible?)
    post(pth, numnodesL, numnodesR, numnodesB,
         numnodesT, b, meshSize, Qp, W, Vf, num_vor_pts)
    # return Vf


if __name__ == '__main__':
    # pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"          ## Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    # pth = "/bigwork/nhkbmort/felix"  # Linux
    pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize = stp()
    # b = 8  8
    abqVor(pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize)
