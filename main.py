from abaqus import *
from abaqusConstants import *
from abq_voronoi import *
from abq_setup import *

# Execution
Mdb()

# call setup
pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize = stp()

# execute main
# for b in range(loop, loop+1):
abqVor(pth, b, W, H, d_min, d_max, strut_width, Qp, meshSize)

# numnodesL, numnodesR = abq(pth, meshSize)
# ## create simulation folder
# dirName = createDir(pth)
# ## execute post
# # post(numnodesL, numnodesR, pth, dirName, meshSize)
