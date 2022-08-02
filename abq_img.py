from abaqus import *
from abaqusConstants import *


def abqImg(pth):
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
     # color coding of materials
     ## filler and matrix
     p = mdb.models['Model-1'].parts['Section_of_CM']
     session.viewports['Viewport: 1'].setValues(displayedObject=p)
     session.viewports['Viewport: 1'].enableMultipleColors()
     session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
     cmap = session.viewports['Viewport: 1'].colorMappings['Material']
     cmap.updateOverrides(overrides={'Material-filler': (True, '#000080', 'Default',
                                                         '#000080'), 'Material-filler': (True, '#0099FF', 'Default', '#0099FF')})
     session.viewports['Viewport: 1'].setColor(colorMapping=cmap)

     # save picture of probe
     session.viewports['Viewport: 1'].view.setValues(
         session.views['Front'])
     p = mdb.models['Model-1'].parts['Section_of_CM']
     session.viewports['Viewport: 1'].setValues(displayedObject=p)
     session.viewports['Viewport: 1'].enableMultipleColors()
     session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
     session.viewports['Viewport: 1'].enableMultipleColors()
     session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
     cmap = session.viewports['Viewport: 1'].colorMappings['Material']
     session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
     session.viewports['Viewport: 1'].disableMultipleColors()

     session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
                                                            engineeringFeatures=ON, mesh=OFF)
     session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
         meshTechnique=OFF)
     p = mdb.models['Model-1'].parts['Section_of_CM']
     session.viewports['Viewport: 1'].setValues(displayedObject=p)
     session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF,
                                                            engineeringFeatures=OFF)
     session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
         referenceRepresentation=ON)
     session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
                                                            engineeringFeatures=ON)
     session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
         referenceRepresentation=OFF)
     session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF,
                                                            engineeringFeatures=OFF, mesh=OFF)
     # session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
     # meshTechnique=ON)
     session.viewports['Viewport: 1'].view.setValues(
         session.views['Front'])
     session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF,
                                                                          legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
     session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
         meshVisibleEdges=FEATURE)
     session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
         meshEdgesInShaded=OFF)
     session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
         geometryEdgesInShaded=OFF)
     session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
         geometrySilhouetteEdges=OFF)
     session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
         highlightMode=NONE)
     session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
         substructureTranslucency=OFF)
     mdb.models['Model-1'].parts['Section_of_CM'].setValues(
         geometryRefinement=EXTRA_FINE)

     session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
                                                            engineeringFeatures=ON, mesh=ON)
     session.pngOptions.setValues(imageSize=SIZE_ON_SCREEN)
     session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)
     session.printToFile(
         fileName=pth + '/Work/materialprobe',
         format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

     session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
         meshVisibleEdges=ALL)
     session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
         meshEdgesInShaded=ON)
     session.printToFile(
         fileName=pth + '/Work/materialprobe+mesh',
         format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))


if __name__ == '__main__':
     pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"  # Windows
     # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
     abqImg(pth)
