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

ff = input()

try:
    loss_history = pickle.load(open(Resultspath + psf[int(ff)], "rb"))

    import numpy as np
    import matplotlib.pyplot as plt


    plt.plot(loss_history)
    plt.show()
except:
    print("Something went wrong.")
