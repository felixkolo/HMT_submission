from abaqus import *
from abaqusConstants import *
def abqMat(i):
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
    # define the material and section of matrix, assign section
    mdb.models['Model-1'].Material(name='Material-matrix')  # Paraffin
    mdb.models['Model-1'].materials['Material-matrix'].Conductivity(table=((0.2, ), )) # W/mK, assumed solid
    mdb.models['Model-1'].materials['Material-matrix'].Elastic(table=((1, 0.3), )) # GPa
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-matrix', 
        material='Material-matrix', thickness=None)
    p = mdb.models['Model-1'].parts['rectangle1']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['rectangle1']
    p.SectionAssignment(region=region, sectionName='Section-matrix', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

    #define the material of particles, create and assign section
    mdb.models['Model-1'].Material(name='Material-filler') # Copper
    mdb.models['Model-1'].materials['Material-filler'].Conductivity(table=((393, ), )) #W/mK
    mdb.models['Model-1'].materials['Material-filler'].Elastic(table=((132, 0.3), )) # GPa
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-filler', 
        material='Material-filler', thickness=None)
    for k in range(0,i):
        p = mdb.models['Model-1'].parts['node%d'%(k+1)]
        f = p.faces
        faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
        region = regionToolset.Region(faces=faces)
        p = mdb.models['Model-1'].parts['node%d'%(k+1)]
        p.SectionAssignment(region=region, sectionName='Section-filler', offset=0.0, 
            offsetType=MIDDLE_SURFACE, offsetField='', 
            thicknessAssignment=FROM_SECTION)

    p = mdb.models['Model-1'].parts['rectangle2']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models['Model-1'].parts['rectangle2']
    p.SectionAssignment(region=region, sectionName='Section-filler', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

if __name__ == '__main__':
    # pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"          ## Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    abqMat(i)