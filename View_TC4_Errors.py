print("Errors of which model?")
ff = input()
print("Which step? (Number or 'all')")
gg = input()

#1100 | 4899


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
    if gg == "all":
        print(Errordict["Step"])
    else:
        if Errordict["Step"] == int(gg):
            Step = Errordict["Step"]
            Old_Loss = Errordict["Old_Loss"]
            New_Loss = Errordict["New_Loss"]
            Real_Labels = Errordict["Labels"]
            Tokens_And_Labels_ATT = Errordict["Paragraph_tokens_and_labels_and_att"]
            Output = Errordict["Output"]


if gg != "all":
    for i in range(len(Tokens_And_Labels_ATT)):
        Tokens = Tokens_And_Labels[i][0]
        Labels = Tokens_And_Labels[i][1]
        Attention = Tokens_And_Labels[i][2]
        Logits = Output.logits[i].sigmoid()
        for j in range(len(Tokens)):
            if Labels[j][3] == float(1) or Labels[j][0] == float(1):
                print(Tokens[j])
                print(Labels[j])
                print(Atttention[j])
                print(Logits[j])
                input()
        print("______________________________________")
        
