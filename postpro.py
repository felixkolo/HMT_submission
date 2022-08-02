def post(pth, numnodesL, numnodesR, numnodesB, numnodesT, b, meshSize, Qp, W, Vf, num_vor_pts):
    import csv
    import numpy as np
    import os

    # pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"
    # numnodesL=39
    # numnodesR=39
    # numnodesB=39
    # numnodesT=39
    # b=8
    # meshSize=0.3
    # Qp=1.0
    # W=10.0
    if numnodesL == numnodesR:
        numnodes = numnodesL
    else:
        numnodes = numnodesL

    # thermal postpro
    # sigma= 0.1                    ## stress

    ######################################################################################
    # postpro for non-csv-files (needed for Abaqus 6.12 because it does not Support CSV-output)
    with open(pth + "/Work/abaqus.rpt") as f:
        line1 = [line for line in f]
    f.close()

    data = []
    for row in enumerate(line1):
        if row[1][0:16] == '           Total':
            data.append(row[1])

    T1 = data[0].split()
    T2 = data[1].split()

    sumT1 = float(T1[1])
    sumT2 = float(T2[1])

    # dU=sumU/numnodes                ## mean nodel displacement
    dT1 = sumT1 / numnodesL  # mean temperature left edge 1. sim
    dT2 = sumT2 / numnodesB  # mean temperature bottom edge 2. sim

    # epsilon=dU/W                    ## calculate strain

    # results to write in file
    lambd1 = (Qp * W) / dT1  # mean thermal conductivity over length
    lambd2 = (Qp * W) / dT2  # mean thermal conductivity over length
    # E=sigma/epsilon                   ## calculate young's modulus

    # write into data
    # write and append into results-file line by line
    f = open(pth + '/Sim/Training/resultsAll.dat', "a+")
    f.write(str(b) + ', ' + str(round(meshSize, 4)) + ', ' + str(round(lambd1, 4)) + ', '
            + str(round(lambd2, 4)) + ', ' + str(Vf) + ', ' + str(num_vor_pts) + '\n')  # each simulation is one line
    f.close()

    # f=open(pth+'/Sim/test/resultstest.dat', "a+")
    f = open(pth + '/Sim/SimFolders/' + str(b) + '/results.dat', "a+")
    f.write(str(round(meshSize, 4)) + ', ' + str(round(lambd1, 4)) + ', ' +
            str(round(lambd2, 4)) + ', ' + str(Vf) + ', ' + str(num_vor_pts) + '\n')  # just one line for each simulation
    f.close()

    print('Simulation #' + str(b) + '\n' + 'Thermal conductivity 1st simulation: ' + str(round(lambd1, 4)) + '\n' +
          'Thermal conductivity 2nd simnulation: ' + str(round(lambd2, 4)) + '\n')


if __name__ == '__main__':
    pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"  # Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    post(pth, numnodesL, numnodesR, numnodesB,
         numnodesT, b, meshSize, Qp, W, Vf, num_vor_pts)

    ######################################################################################
    # postpro for csv-files
    # with open(pth+"/Work/abaqus.csv") as f:
    #    reader = csv.reader(f, delimiter=",")
    #    i=0
    #    column1 = [row[11] for row in reader]            ##list of temperatures and displacements
    #
    # T=column1[1:numnodes+1]
    # for i in range(len(T)):
    #    T[i]=T[i].strip()       ## getting rid of white spaces
    #    T[i]=float(T[i])        ## convert to float to be able to calculate
    #
    # mechanical postpro
    # sigma= 0.05   #0.1#stress
    #
    # Du1=column1[numnodes+2:]
    # for i in range(len(Du1)):
    #    Du1[i]=Du1[i].strip()
    #    Du1[i]=float(Du1[i])
    #
    # du1=np.mean(Du1)
    # du1=abs(du1)
    #
    # epsilon=du1/L
