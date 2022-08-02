from abaqus import *
from abaqusConstants import *


def abqMesh(meshSize, W, edges1, edges2):
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
    p = mdb.models['Model-1'].parts['Section_of_CM']
    p.seedPart(size=meshSize, deviationFactor=0.1,
               minSizeFactor=0.1)  # mesh size
    p = mdb.models['Model-1'].parts['Section_of_CM']
    elemType1 = mesh.ElemType(elemCode=DC2D4, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=DC2D3, elemLibrary=STANDARD)
    p = mdb.models['Model-1'].parts['Section_of_CM']
    f = p.faces
    faces = f[0:len(f)]
    pickedRegions = (faces, )
    # p.setMeshControls(regions=f, algorithm=MEDIAL_AXIS)       ## produces mesh too fine for stud. ver.
    p.setMeshControls(regions=f, elemShape=TRI)
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    p.generateMesh()

    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF,
                                                               bcs=OFF, predefinedFields=OFF, connectors=OFF, optimizationTasks=OFF,
                                                               geometricRestrictions=OFF, stopConditions=OFF)

    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)

    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])

    # create node sets
    e1 = a.instances['Section_of_CM-1'].edges

    # find nodes from right boundary
    elist2 = []
    k = 0
    for i in range(0, len(e1)):
        if e1[i].pointOn[0][0] == (float(W) + float(W) / 4):
            elist2.append(e1[i].pointOn)
            if len(elist2) == 1:
                edges2 = e1.findAt(elist2[k])
                k = k + 1
            else:
                edges2 = edges2 + e1.findAt(elist2[k])
                k = k + 1
        else:
            i = i + 1

    for i in range(0, len(edges2)):
        if i == 0:
            nr = edges2[i].getNodes()
        else:
            nr = nr + edges2[i].getNodes()

    # Create Set of Nodes of right boundary
    sr = a.Set(nodes=nr, name='Right Nodes')
    numnodesR = len(sr.nodes)  # Number of nodes on Left Side

    # find nodes from left boundary
    elist1 = []
    k = 0
    for i in range(0, len(e1)):
        if e1[i].pointOn[0][0] == (float(W) / 4):
            elist1.append(e1[i].pointOn)
            if len(elist1) == 1:
                edges1 = e1.findAt(elist1[k])
                k = k + 1
            else:
                edges1 = edges1 + e1.findAt(elist1[k])
                k = k + 1
        else:
            i = i + 1

    for i in range(0, len(edges1)):
        if i == 0:
            nl = edges1[i].getNodes()
        else:
            nl = nl + edges1[i].getNodes()

    # Create Set of Nodes of left boundary
    sl = a.Set(nodes=nl, name='Left Nodes')
    numnodesL = len(sl.nodes)  # Number of nodes on Left Side

    return numnodesL, numnodesR


if __name__ == '__main__':
    # pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"          ## Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    numnodesL, numnodesR = abqMesh(meshSize, W)
