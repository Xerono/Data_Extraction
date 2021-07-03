import os
import pickle

CurDir = os.getcwd()


Resultspath = CurDir + "/Results/"
Files = os.listdir(Resultspath)
for file in Files:
    if "TC1e_History_" in file:
        File = file
loss_history = pickle.load(open(Resultspath + File, "rb"))
Used_Time = float(File.split("_")[2].split(".pickle")[0])

