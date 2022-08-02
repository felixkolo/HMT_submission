# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:03:31 2020

@author: Felix
"""

# Plot thermal conductivity over volume fraction
import csv
import sys
import os.path
# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
print(sys.executable)
print(sys.version)

pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"
Vf = []
h_h = []
h_v = []
h_hv = []
for i in range(23,26):
    fullpath = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit/vf-var-geometry/vf-var-geom/"+str(i)+"/Sim/Training/resultsAll.dat"
    with open(fullpath) as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        alles = [row[2:6] for row in reader]
        # numvorpts = [row[5] for row in reader]

    for i, line in enumerate(alles):
        Vf.append(float(line[2])*100)
        h_h.append(float(line[0]))
        h_v.append(float(line[1]))
        h_hv.append((float(line[0])+float(line[1]))/2)

#%% plot thermal conductivity over volume fraction (porosity)
plt.plot(Vf, h_hv, '+', color='black')
plt.xlabel(r'$\rightarrow$'+' Porosity '+r'$\epsilon$'+' of Cu foam in composite [%]')
plt.ylabel(r'$\rightarrow$'+' Thermal Conductivity (horizontal) '+r'$[W/m^{2}]$')
# plt.xlim(0.84, 0.92)
# plt.title('Histogram of porosity of samples')
# plt.legend()
plt.savefig(os.path.join(pth, 'Sim', 'Evaluation', 'plot-h_hv-vf.png'), dpi=300)
plt.show()