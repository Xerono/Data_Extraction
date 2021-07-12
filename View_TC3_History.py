import os
import pickle

CurDir = os.getcwd()


Resultspath = CurDir + "/Results/"
Files = os.listdir(Resultspath)
psf = []
for file in Files:
    if "TC3" in file:
        psf.append(file)

print("Choose number for which file to display:")
for i in range(len(psf)):
    print(str(i) + ": " + psf[i])

Modelspath = CurDir + "/Models/"

ff = input()

try:
    loss_history = pickle.load(open(Resultspath + psf[int(ff)], "rb"))
    Code = psf[int(ff)].split("_")[1]
    Paramsfile = "TC3_" + Code + "_Model_Coordinates/Parameters.pickle"
    print(Paramsfile)
    ParaFile = Modelspath + Paramsfile
    Params = pickle.load(open(ParaFile, "rb"))

    import numpy as np
    import matplotlib.pyplot as plt

    plt.title(Code + " - " + str(Params["FullTime"]) + " (Max 180000)")
    plt.plot(loss_history)
    plt.show()
except:
    print("Something went wrong.")
