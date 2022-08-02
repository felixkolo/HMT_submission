def moveFiles(pth, b):
    import os
    import shutil
    os.chdir(pth + "/Sim/")
    picDir = "Training/pic"
    if not os.path.exists(picDir):
        os.makedirs(picDir)
        print("Directory ", picDir,  " Created ")
    workdir = os.path.join(pth, 'Work')
    destdir = os.path.join(pth, 'Sim', 'SimFolders', str(b))

    #    ## unix
    #    # os.popen('cp '+pathtocurrentfile+'Job-1.odb '+pathtodestination+'Job-1.odb')
    #    # os.popen('cp '+pathtocurrentfile+'Job-2.odb '+pathtodestination+'Job-2.odb')
    #    # os.popen('cp '+pathtocurrentfile+'Job-1.inp '+pathtodestination+'Job-1.inp')
    #    # os.popen('cp '+pathtocurrentfile+'Job-2.inp '+pathtodestination+'Job-2.inp')
    #    # os.popen('cp '+pathtocurrentfile+'abaqus.rpt '+pathtodestination+'abaqus.rpt')
    #    # os.popen('cp '+pathtocurrentfile+'materialprobe.png '+pathtodestination+'materialprobe.png')
    #    # os.popen('cp '+pathtocurrentfile+'materialprobe.png '+pth+'/Sim/pic/materialprobe'+dirName+'.png')
    #    # os.popen('cp '+pathtocurrentfile+'model.cae '+pathtodestination+'model.cae')
    # windows
    # shutil.copyfile(''+pathtocurrentfile+'Job-1.odb', ''+pathtodestination+'Job-1.odb')
    # shutil.copyfile(''+pathtocurrentfile+'Job-2.odb', ''+pathtodestination+'Job-2.odb')
    # shutil.copyfile('' + os.path.join(workdir, 'Job-1.inp'),
    #                 '' + os.path.join(destdir, 'Job-1.inp'))
    # shutil.copyfile('' + os.path.join(workdir, 'Job-2.inp'),
    #                 '' + os.path.join(destdir, 'Job-2.inp'))
    # shutil.copyfile('' + os.path.join(workdir, 'abaqus.rpt'),
    #                 '' + os.path.join(destdir, 'abaqus.rpt'))
    # shutil.copyfile('' + os.path.join(workdir, 'materialprobe.png'),
    #                 '' + os.path.join(destdir, 'materialprobe.png'))
    # shutil.copyfile('' + os.path.join(workdir, 'materialprobe+mesh.png'),
    #                 '' + os.path.join(destdir, 'materialprobe+mesh.png'))
    # shutil.copyfile('' + os.path.join(workdir, 'model.cae'),
    #                 '' + os.path.join(destdir, 'model.cae'))
    shutil.copyfile('' + os.path.join(workdir, 'materialprobe.png'), '' +
                    os.path.join(pth, 'Sim', 'Training', 'pic', 'materialprobe' + str(b) + '.png'))
    # shutil.copyfile('' + os.path.join(workdir, 'materialprobe+mesh.png'), '' + os.path.join(
    #     pth, 'Sim', 'Training', 'pic', 'withMesh', 'materialprobe' + str(b) + '+mesh.png'))

    # Mdb()
if __name__ == '__main__':
    pth = "D:/Dokumente/Uni/Masterarbeit/Masterarbeit"  # Windows
    # pth = "/home/hiwibm03/Downloads/Masterarbeit"             ##Linux
    moveFiles(pth, b=5)
