from abaqus import *
from abaqusConstants import *
import math


def createPart(pth, W, H, vor_vertices, vor_ridges, d_min, d_max, strut_width, num_vor_pts):
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

    # create a rectangle as Matrix
    for i in range(1, 3):
        s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                     sheetSize=20.0)
        g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
        s1.setPrimaryObject(option=STANDALONE)
        s1.rectangle(point1=(W / 4, H / 4), point2=(W + W / 4, H + H / 4))
        # s1.rectangle(point1=(0, 0), point2=(W, H))
        p = mdb.models['Model-1'].Part(name='rectangle%d' % (i), dimensionality=TWO_D_PLANAR,
                                       type=DEFORMABLE_BODY)
        p = mdb.models['Model-1'].parts['rectangle%d' % (i)]
        p.BaseShell(sketch=s1)
        s1.unsetPrimaryObject()
        p = mdb.models['Model-1'].parts['rectangle%d' % (i)]
        session.viewports['Viewport: 1'].setValues(displayedObject=p)
        del mdb.models['Model-1'].sketches['__profile__']

    # create nodes
    i = 0
    for row in vor_vertices:
        # if W-d_max > row[0] > d_max and H-d_max > row[1] > d_max:
        i = i + 1
        s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                     sheetSize=20.0)
        g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
        s1.setPrimaryObject(option=STANDALONE)
        s1.CircleByCenterPerimeter(center=(row[0], row[1]), point1=(
            row[0] + 0.5 * d_max, row[1]))
        # s1.CircleByCenterPerimeter(center=(row[0], row[1]), point1=(
        #     row[0] + 0.075 / 2, row[1]))
        p = mdb.models['Model-1'].Part(name='node%d' % (i), dimensionality=TWO_D_PLANAR,
                                       type=DEFORMABLE_BODY)
        p = mdb.models['Model-1'].parts['node%d' % (i)]
        p.BaseShell(sketch=s1)
        s1.unsetPrimaryObject()
        p = mdb.models['Model-1'].parts['node%d' % (i)]
        session.viewports['Viewport: 1'].setValues(displayedObject=p)
        del mdb.models['Model-1'].sketches['__profile__']
    else:
        next

    # # create struts
    j = 0
    for vpair in vor_ridges:
        if vpair[0] >= 0 and vpair[1] >= 0:
            j = j + 1
            v0 = vor_vertices[vpair[0]]
            v1 = vor_vertices[vpair[1]]

            # debug
            # failure bei j=68, 76
            # for j in range(74, 100):
            # i = 0
            # j = 2
            # v0 = vor_vertices[vor_ridges[j][0]]
            # v1 = vor_vertices[vor_ridges[j][1]]
            #
            vor_points = []
            strut_mean = []
            csvfile = np.loadtxt(os.path.join(
                pth, 'Sim', 'Evaluation', 'strut_mean.txt'))
            for row in csvfile:
                vor_points.append(row[0])
                strut_mean.append(row[1])

            polyreg = np.poly1d(np.polyfit(vor_points, strut_mean, 3))
            # equation regression: -3.808e-07 x^3 + 0.0001675 x^2 - 0.02693 x + 2.34
            strut_mean = polyreg(num_vor_pts)  # mean strut length
            strut_length = math.sqrt(
                pow(v0[0] - v1[0], 2) + pow(v0[1] - v1[1], 2))

            # calculate strut width
            # radius (own (bad) Regression) at mid-span
            strut_width = sqrt(
                (0.000011297777 * (num_vor_pts - 180)**2 + 0.0418) / pi)
            strut_norm = strut_length / strut_mean
            # A0 = (strut_width / 2)**2 * pi / 3.5    # mean cross section in the middle of a strut
            # mean cross section in the middle of a strut ## CORRECT WAY
            A0 = (strut_width)**2 * pi
            strut_cross = (0.6633 + 0.2684 * strut_norm **
                           (-2.5963)) * A0  # from Jang et al. 2008: actual cross section of individual strut
            if strut_cross > 2 * A0:
                strut_cross = 2 * A0

            As = 3.5 * strut_cross  # strut cross section at end of strut
            # scale = 1/(0.5*4*3.5)   # This reproduces the old geometry
            scale = 0.166666666666666666666666   # scaling factor to meet volume fraction
            rs = sqrt(As / pi) * scale  # radius of strut at end of strut
            dy = (v1[1] - v0[1])
            # point1 = (v0[0] - 0.5 * rs, v0[1])
            # point2 = (v0[0] + 0.5 * rs, v0[1] + strut_length)
            point1 = (v0[0] - rs, v0[1])  # CORRECT WAY
            point2 = (v0[0] + rs, v0[1] + strut_length)  # CORRECT WAY
            if v1[0] < v0[0] and v1[1] > v0[1] or v1[0] < v0[0] and v1[1] < v0[1]:
                alpha = np.arccos(dy / strut_length) * 180 / math.pi
            elif v1[0] > v0[0] and v1[1] < v0[1] or v1[0] > v0[0] and v1[1] > v0[1]:
                alpha = 360 - np.arccos(dy / strut_length) * 180 / math.pi

            # create rectangles and rotate them to connect nodes from v0 to v1.
            s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                        sheetSize=200.0)
            g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
            s.setPrimaryObject(option=STANDALONE)
            s.rectangle(point1=point1, point2=point2)
            s.delete(objectList=(
                g.findAt((round(v0[0], 7), round(v0[1], 7))), ))
            s.delete(objectList=(
                g.findAt((round(v0[0], 7), round(v0[1] + strut_length, 7))), ))
            s.CircleByCenterPerimeter(center=(v0[0], v0[1]), point1=(
                # v0[0] + 0.5 * rs, v0[1]))
                v0[0] + rs, v0[1]))  # CORRECT WAY
            # s.autoTrimCurve(curve1=g.findAt((v0[0], v0[1] + 0.5 * rs)), point1=(
            # v0[0], v0[1] + 0.5 * rs))
            s.autoTrimCurve(curve1=g.findAt((v0[0], v0[1] + rs)), point1=(  # CORRECT WAY
                v0[0], v0[1] + rs))  # CORRECT WAY
            s.CircleByCenterPerimeter(center=(
                # v0[0], v0[1] + strut_length), point1=(v0[0] + 0.5 * rs, v0[1] + strut_length))
                v0[0], v0[1] + strut_length), point1=(v0[0] + rs, v0[1] + strut_length))  # CORRECT WAY
            # s.autoTrimCurve(curve1=g.findAt((v0[0], v0[1] + strut_length - 0.5 * rs)), point1=(
            #     v0[0], v0[1] + strut_length - 0.5 * rs))
            s.autoTrimCurve(curve1=g.findAt((v0[0], v0[1] + strut_length - rs)), point1=(  # CORRECT WAY
                v0[0], v0[1] + strut_length - rs))  # CORRECT WAY
            if strut_norm < 0.6:
                glist = [g[2], g[4], g[7], g[9]]
            else:
                # einschnuerung machen
                r = []
                yp = []
                pl = []
                pr = []
                A = []
                for k in range(11):
                    yp.append(0.1 * k * strut_length)
                    # A.append(strut_cross*(36 * (yp[-1] / strut_length - 0.5)**4 + (yp[-1] / strut_length - 0.5)**2 + 1))
                    # r.append(sqrt(A[-1]/pi))
                    r.append(rs * sqrt(((1 / 3.5)) *
                                       (36 * (yp[-1] / strut_length - 0.5)**4 + (yp[-1] / strut_length - 0.5)**2 + 1)))
                    pl.append([v0[0] - r[-1], v0[1] + yp[-1]])
                    pr.append([v0[0] + r[-1], v0[1] + yp[-1]])
                    if k > 0:
                        s.Line(point1=(pl[-1]), point2=(pl[-2]))
                        s.Line(point1=(pr[-1]), point2=(pr[-2]))

                # s.delete(objectList=(
                #     g.findAt((round(v0[0] - 0.5 * rs, 7), round(v0[1] + 0.001, 7))), ))
                # s.delete(objectList=(
                #     g.findAt((round(v0[0] + 0.5 * rs, 7), round(v0[1] + 0.001, 7))), ))
                s.delete(objectList=(
                    g.findAt((round(v0[0] - rs, 7), round(v0[1] + 0.001, 7))), ))  # CORRECT WAY
                s.delete(objectList=(
                    g.findAt((round(v0[0] + rs, 7), round(v0[1] + 0.001, 7))), ))  # CORRECT WAY

                glist = []
                for n in range(7, len(g) + 8):
                    if n != 8:
                        glist.append(g[n])

            s.rotate(centerPoint=(v0[0], v0[1]),
                     angle=alpha, objectList=(glist))
            p = mdb.models['Model-1'].Part(name='strut%d' % (j), dimensionality=TWO_D_PLANAR,
                                           type=DEFORMABLE_BODY)
            p = mdb.models['Model-1'].parts['strut%d' % (j)]
            p.BaseShell(sketch=s)
            s.unsetPrimaryObject()
            p = mdb.models['Model-1'].parts['strut%d' % (j)]
            session.viewports['Viewport: 1'].setValues(displayedObject=p)
            del mdb.models['Model-1'].sketches['__profile__']
    return i, j


if __name__ == '__main__':
    # pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"          ## Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    i, j = createPart(pth, W, H, vor_vertices, vor_ridges,
                      d_min, d_max, strut_width, num_vor_pts)
