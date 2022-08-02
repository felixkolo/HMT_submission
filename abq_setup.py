import __main__
import numpy as np
import os.path


def stp():
    # pth = "/bigwork/nhkbmort/felix"  # Linux
    pth = 'D:/Dokumente/Uni/Masterarbeit/Masterarbeit'  # Windows
    for i in range(1, 10000):
        if not os.path.exists(os.path.join(pth, 'Sim', 'SimFolders', str(i))):
            b = i - 1
            break

    W = 10.0
    H = 10.0
    meshSize = 0.1
    strut_width = 0.1
    d_min = 0.2 * strut_width
    d_max = 0.2 * strut_width
    Qp = 1.0  # surface heat flux: W/m^2
    return pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize


if __name__ == '__main__':
    pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"  # Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    # pth = "/bigwork/nhkbmort/felix"                     ## Linux
    pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize = stp()
