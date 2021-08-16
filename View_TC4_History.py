import os
import pickle

CurDir = os.getcwd()

while True:
    Resultspath = CurDir + "/Results/"
    Files = os.listdir(Resultspath + "TC4_Loss")
    psf = []
    for file in Files:
        psf.append(file)

    print("Choose number for which file to display:")
    for i in range(len(psf)):
        print(str(i) + ": " + psf[i])

    Modelspath = CurDir + "/Models/"

    ff = input()


    File = psf[int(ff)]
    Code = File.split(".")[0]
    loss_history = pickle.load(open(Resultspath + "TC4_Loss/" + File, "rb"))

    Paramsfile = "TC4_" + Code + "_Model/Parameters.pickle"
    ParaFile = Modelspath + Paramsfile
    Params = pickle.load(open(ParaFile, "rb"))


    import numpy as np
    import matplotlib.pyplot as plt
    plt.title(Code + " - " + str(Params["FullTime"]) + " (Max 288000)")
    plt.yscale('log')
    plt.plot(loss_history)
    plt.show()

