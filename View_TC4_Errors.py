print("Errors of which model?")
ff = input()

import pickle

import os


CurDir  = os.getcwd()


Errorpath = CurDir + "/Results/TC4_Errors/"

try:
    Errors = pickle.load(open(Errorpath + ff + ".pickle", "rb"))
except:
    import sys
    print("Error loading file")
    sys.exit()



for Errordict in Errors:
    for ekey in Errordict.keys():
        print(ekey + ":")
        print(Errordict[ekey])
    input()
