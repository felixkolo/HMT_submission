from abaqus import *
from abaqusConstants import *


def abqAss(i, j):
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
     import math
     import numpy as np
     import random
     import sys
     import os
     # import all nodes and struts and merge them into a whole part
     a = mdb.models['Model-1'].rootAssembly
     session.viewports['Viewport: 1'].setValues(displayedObject=a)
     assembl = ()
     a1 = mdb.models['Model-1'].rootAssembly
     a1.DatumCsysByDefault(CARTESIAN)

     for m in range(1, i + 1):
          p = mdb.models['Model-1'].parts['node%d' % (m)]
          a1.Instance(name='node%d-1' % (m), part=p, dependent=ON)
          assembl = assembl + (a1.instances['node%d-1' % (m)], )

     for m in range(1, j + 1):
          p = mdb.models['Model-1'].parts['strut%d' % (m)]
          a1.Instance(name='strut%d-1' % (m), part=p, dependent=ON)
          assembl = assembl + (a1.instances['strut%d-1' % (m)], )

     a1 = mdb.models['Model-1'].rootAssembly
     a1.InstanceFromBooleanMerge(name='Set_of_nodes', instances=(assembl),
                                 keepIntersections=OFF, originalInstances=SUPPRESS, domain=GEOMETRY)

     # import the square,and cut the nodes regions, get the matrix
     a1 = mdb.models['Model-1'].rootAssembly
     p = mdb.models['Model-1'].parts['rectangle1']
     a1.Instance(name='rectangle1-1', part=p, dependent=ON)
     a1 = mdb.models['Model-1'].rootAssembly
     a1.InstanceFromBooleanCut(name='Matrix',
                               instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['rectangle1-1'],
                               cuttingInstances=(
                                    a1.instances['Set_of_nodes-1'], ),
                               originalInstances=SUPPRESS)
     p = mdb.models['Model-1'].parts['Matrix']
     Vf = p.getArea(p.faces) / 100
     a1 = mdb.models['Model-1'].rootAssembly
     p = mdb.models['Model-1'].parts['rectangle2']
     a1.Instance(name='rectangle2-1', part=p, dependent=ON)
     a1.InstanceFromBooleanCut(name='rect_filler',
                               instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['rectangle2-1'],
                               cuttingInstances=(
                                    a1.instances['Matrix-1'], ),
                               originalInstances=SUPPRESS)
     a1.features['Matrix-1'].resume()

     # merge the matrix and all particles
     a1 = mdb.models['Model-1'].rootAssembly
     a1.InstanceFromBooleanMerge(name='Section_of_CM', instances=(a1.instances['Matrix-1'],
                                                                  a1.instances['rect_filler-1'], ), keepIntersections=ON,
                                 originalInstances=SUPPRESS, domain=GEOMETRY)
     session.viewports['Viewport: 1'].assemblyDisplay.setValues(
         adaptiveMeshConstraints=ON)
     return Vf


if __name__ == '__main__':
     # pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"          ## Windows
     # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
     Vf = abqAss(i, j)
