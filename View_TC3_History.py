import os
import pickle

CurDir = os.getcwd()


Resultspath = CurDir + "/Results/"
Files = os.listdir(Resultspath + "TC3_Loss")
psf = []
for file in Files:
    psf.append(file)

print("Choose number for which file to display:")
for i in range(len(psf)):
    print(str(i) + ": " + psf[i])

Modelspath = CurDir + "/Models/"

ff = input()

try:
    File = psf[int(ff)]
    Code = File.split(".")[0]
    loss_history = pickle.load(open(Resultspath + "TC3_Loss/" + File, "rb"))
    if list(Code)[3] == "1":
        closs_history = pickle.load(open(Resultspath + "TC3_Custom_Loss/" + File, "rb"))
    Paramsfile = "TC3_" + Code + "_Model/Parameters.pickle"
    ParaFile = Modelspath + Paramsfile
    Params = pickle.load(open(ParaFile, "rb"))

    import numpy as np
    import matplotlib.pyplot as plt
    plt.title(Code + " - " + str(Params["FullTime"]) + " (Max 288000)")
    plt.yscale('log')
    plt.plot(loss_history)
    if list(Code)[3] == "1":
        plt.plot(closs_history, "r-")
    plt.show()
except:
    print("Something went wrong.")
