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
    Code = psf[int(ff)]
    loss_history = pickle.load(open(Resultspath + "TC3_Loss" + Code, "rb"))
    if list(Code][3] == "1":
            closs_history = pickle.load(open(Resultspath + "TC3_Custom_Loss/" + Code, "rb"))
    Paramsfile = "TC3_" + Code + "_Model_Coordinates/Parameters.pickle"
    
    ParaFile = Modelspath + Paramsfile
    Params = pickle.load(open(ParaFile, "rb"))

    import numpy as np
    import matplotlib.pyplot as plt

    plt.title(Code + " - " + str(Params["FullTime"]) + " (Max 180000)")
    plt.plot(loss_history)
    if list(Code][3] == "1":
        plt.plot(closs_history)
    plt.show()
except:
    print("Something went wrong.")
