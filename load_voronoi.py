import numpy as np
import os.path
def loadVor(pth, b):

    # load vertices
    vor_vertices = np.loadtxt(os.path.join(
        pth, 'Sim', 'SimFolders', str(b), 'voronoi_vertices.txt'), skiprows=1)
    vor_vertices = vor_vertices.reshape(int(len(vor_vertices) / 2), 2)

    # load ridges
    vor_ridges = np.loadtxt(os.path.join(
        pth, 'Sim', 'SimFolders', str(b), 'voronoi_ridges.txt'), skiprows=1)
    vor_ridges = vor_ridges.reshape(int(len(vor_ridges) / 2), 2)

    # load number of voronoi points
    num_vor_pts = np.loadtxt(os.path.join(
        pth, 'Sim', 'SimFolders', str(b), 'num_voronoi_seeds.txt'), skiprows=1)

    return vor_vertices, vor_ridges, num_vor_pts


if __name__ == '__main__':
    pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"  # Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    vor_vertices, vor_ridges = loadVor(pth, b=1)
