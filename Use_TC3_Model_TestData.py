import os
import pickle

Debug = False


CurDir = os.getcwd()


import sqlite3

Maxlength = 917

import Module_Coordinates as mc

                          

DatasetTest = []
NumbersTest = [0,0,0]
TestData = pickle.load(open(CurDir + "/Files/TC3_Test.pickle", "rb"))
for (Par, ListOfCoordsInPar) in TestData:
    for (RealCoords, StringCoords) in ListOfCoordsInPar:
        DatasetTest.append((RealCoords, len(RealCoords), mc.split_string(Par)))
        if len(RealCoords) == 6:
            NumbersTest[0] += 1
        else:
            if len(RealCoords) == 8:
                NumbersTest[1] += 1
            else:
                NumbersTest[2] += 1

DatasetTrain = []
NumbersTrain = [0,0,0]

TrainData = pickle.load(open(CurDir + "/Files/TC3_Training.pickle", "rb"))
for (Par, ListOfCoordsInPar) in TrainData:
    for (RealCoords, StringCoords) in ListOfCoordsInPar:
        DatasetTrain.append((RealCoords, len(RealCoords), mc.split_string(Par)))
        if len(RealCoords) == 6:
            NumbersTrain[0] += 1
        else:
            if len(RealCoords) == 8:
                NumbersTrain[1] += 1
            else:
                NumbersTrain[2] += 1
Datasets = [(DatasetTest, NumbersTest, True), (DatasetTrain, NumbersTrain, False)]

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PreTrainedModel = 'bert-base-cased'
from transformers import BertTokenizerFast
Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)




ModPath = CurDir + "/Models/"

All_Models = os.listdir(ModPath)

Models = []
from transformers import BertForTokenClassification
for mdl in All_Models:
    if "TC3" in mdl:
        Models.append(mdl)

for mdl in Models:
    for (Dataset, Numbers, IsThisTestData) in Datasets:
        Paras = mdl.split("_")[1]

        Cut_Par = bool(int(Paras[0]))
        CTN = bool(int(Paras[1]))
        Dele = bool(int(Paras[2]))
        CLoss = bool(int(Paras[3]))
        DLabels = bool(int(Paras[4]))
        Model_Path = ModPath + mdl
        
        if DLabels:
            model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=11).to(device)
            Rel_Labels = [1,2,3,4,5,7,8,9]
        else:
            model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
            Rel_Labels = [1,2,3,4,5]
        model.eval()



        Resultsdict = {}
        o1 = "NullEins"
        o0 = "NullNull"
        B1F = "B1SoilFound"
        B1N = "B1SoilNotF"
        B0F = "B0SoilFound"
        B0N = "B0SoilNotF"
        Resultsdict[11] = 0 # Bewertet 1, Label 1
        Resultsdict[10] = 0 # Bewertet 1, Label 0
        Resultsdict[o1] = 0
        Resultsdict[o0] = 0
        Resultsdict[B1F] = 0
        Resultsdict[B1N] = 0
        Resultsdict[B0F] = 0
        Resultsdict[B0N] = 0

        LabelDict = {}
        for i in range(1,10):
            LabelDict[i] = 0

        Runner = 0
        HitDict = {}
        for i in range(9):
            HitDict[i] = 0

        for (PotCords, LenCoords, SplitPar) in Dataset:
            if len(SplitPar)<Maxlength:
                Tokens = Tokenizer.tokenize(SplitPar)
                Labels = mc.get_label(SplitPar, model, Tokenizer)
                Classes = mc.get_token_class(Tokens, Labels)
                for ClassesPerToken in Classes:
                    for Label in ClassesPerToken:
                        if Label in LabelDict.keys():
                            LabelDict[Label] += 1
                RevTokens = mc.extract_relevant_classes(Tokens, Classes, Rel_Labels)

                FoundRele = False
                for (Token, Labellist) in RevTokens:
                    for lbl in Labellist:
                        FoundRele = True

                if LenCoords == 0:
                    if FoundRele:
                        Resultsdict[10] += 1
                    else:
                        Resultsdict[o0] += 1
                else:
                    if FoundRele:
                        Resultsdict[11] += 1
                    else:
                        Resultsdict[o1] += 1

                    (GradE, MinE, SekE, DirE, Grad2E, Min2E, Sek2E, rnum) = mc.ToCoords(RevTokens)
                    ReturnCoords = []
                    for i in range(LenCoords):
                        ReturnCoords.append(False)
                    if not DLabels:
                        if LenCoords == 8:
                            for grad in GradE:
                                if (not ReturnCoords[0]) and grad == PotCords[0]:
                                    ReturnCoords[0] = grad
                                if (not ReturnCoords[4]) and grad == PotCords[4]:
                                    ReturnCoords[4] = grad
                            for mint in MinE:
                                if (not ReturnCoords[1]) and mint == PotCords[1]:
                                    ReturnCoords[1] = mint
                                if (not ReturnCoords[5]) and mint == PotCords[5]:
                                    ReturnCoords[5] = mint
                            for sek in SekE:
                                if (not ReturnCoords[2]) and sek == PotCords[2]:
                                    ReturnCoords[2] = sek
                                if (not ReturnCoords[6]) and sek == PotCords[6]:
                                    ReturnCoords[6] = sek
                            for dire in DirE:
                                diru = dire.upper()
                                if (not ReturnCoords[3]) and diru == PotCords[3]:
                                    ReturnCoords[3] = diru
                                if (not ReturnCoords[7]) and diru == PotCords[7]:
                                    ReturnCoords[7] = diru
                        else:
                            for grad in GradE:
                                if (not ReturnCoords[0]) and grad == PotCords[0]:
                                    ReturnCoords[0] = grad
                                if (not ReturnCoords[3]) and grad == PotCords[3]:
                                    ReturnCoords[3] = grad
                            for mint in MinE:
                                if (not ReturnCoords[1]) and mint == PotCords[1]:
                                    ReturnCoords[1] = mint
                                if (not ReturnCoords[4]) and mint == PotCords[4]:
                                    ReturnCoords[4] = mint
                            for dire in DirE:
                                diru = dire.upper()
                                if (not ReturnCoords[2]) and diru == PotCords[2]:
                                    ReturnCoords[2] = diru
                                if (not ReturnCoords[5]) and diru == PotCords[5]:
                                    ReturnCoords[5] = diru
                    else:
                        if LenCoords == 8:
                            for grad in GradE:
                                if (not ReturnCoords[0]) and grad == PotCords[0]:
                                    ReturnCoords[0] = grad
                            for grad in Grad2E:
                                if (not ReturnCoords[4]) and grad == PotCords[4]:
                                    ReturnCoords[4] = grad
                            for mint in MinE:
                                if (not ReturnCoords[1]) and mint == PotCords[1]:
                                    ReturnCoords[1] = mint
                            for mint in Min2E:
                                if (not ReturnCoords[5]) and mint == PotCords[5]:
                                    ReturnCoords[5] = mint
                            for sek in SekE:
                                if (not ReturnCoords[2]) and sek == PotCords[2]:
                                    ReturnCoords[2] = sek
                            for sek in Sek2E:
                                if (not ReturnCoords[6]) and sek == PotCords[6]:
                                    ReturnCoords[6] = sek
                            for dire in DirE:
                                diru = dire.upper()
                                if (not ReturnCoords[3]) and diru == PotCords[3]:
                                    ReturnCoords[3] = diru
                                if (not ReturnCoords[7]) and diru == PotCords[7]:
                                    ReturnCoords[7] = diru
                        else:
                            for grad in GradE:
                                if (not ReturnCoords[0]) and grad == PotCords[0]:
                                    ReturnCoords[0] = grad
                            for grad in Grad2E:
                                if (not ReturnCoords[3]) and grad == PotCords[3]:
                                    ReturnCoords[3] = grad
                            for mint in MinE:
                                if (not ReturnCoords[1]) and mint == PotCords[1]:
                                    ReturnCoords[1] = mint
                            for mint in Min2E:
                                if (not ReturnCoords[4]) and mint == PotCords[4]:
                                    ReturnCoords[4] = mint
                            for dire in DirE:
                                diru = dire.upper()
                                if (not ReturnCoords[2]) and diru == PotCords[2]:
                                    ReturnCoords[2] = diru
                                if (not ReturnCoords[5]) and diru == PotCords[5]:
                                    ReturnCoords[5] = diru
                    Hits = 0
                    Found = True
                    for i in range(len(ReturnCoords)):
                        if ReturnCoords[i] == PotCords[i]:
                            Hits += 1
                        else:
                            Found = False
                    HitDict[Hits] += 1
                    if Found:
                        Resultsdict[B1F] += 1               
                    else:
                        if FoundRele:
                            Resultsdict[B1N] += 1
                        else:
                            Resultsdict[B0N] += 1
               
                if Debug and FoundRele:
                    for (Token, Labellist) in RevTokens:
                        for lbl in Labellist:
                            print(Token + " | " + str(IntLabelDict[lbl]))
                    print(PotCords)
                    print(ReturnCoords)
                    print(SplitPar)
                    input()
                if Runner%10000==0:
                    print(Paras + " - " + str(Runner) + "/" + str(len(Dataset)))
                Runner+=1
                    


                
        results_list = []
        if (Resultsdict[11] + Resultsdict[10]) == 0:
            prec = 0
        else:
            prec = Resultsdict[11]/(Resultsdict[11] + Resultsdict[10])
        if (Resultsdict[11] + Resultsdict[o1]) == 0:
            rec = 0
        else:
            rec = Resultsdict[11]/(Resultsdict[11] + Resultsdict[o1])
        if prec + rec == 0:
            fval = 0
        else:
            fval = (2*prec*rec)/(prec+rec) 
        results_list.append((int(Cut_Par), int(CTN), int(Dele), int(CLoss), int(DLabels), int(IsThisTestData),
                             Resultsdict[11], Resultsdict[10], Resultsdict[o1], Resultsdict[o0],
                             Resultsdict[B1F], Resultsdict[B1N], Resultsdict[B0F], Resultsdict[B0N],
                             HitDict[0], HitDict[1], HitDict[2], HitDict[3], HitDict[4], HitDict[5], HitDict[6], HitDict[7], HitDict[8],
                             Numbers[0], Numbers[1], Numbers[2],
                             LabelDict[1], LabelDict[2], LabelDict[3], LabelDict[4], LabelDict[5], LabelDict[6], LabelDict[7], LabelDict[8], LabelDict[9],
                             prec, rec, fval
                            ))


        ResDatabase = CurDir + "/Results/Results.db"

        Con = sqlite3.connect(ResDatabase)
        Cur = Con.cursor()

        sql_command = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='TC3_Test'"
        res = Cur.execute(sql_command).fetchall()
        if res[0][0] == 0:
            sql_command = """
                    CREATE TABLE TC3_Test (
                    CutPar INTEGER NOT NULL,
                    CTN INTEGER NOT NULL,
                    Dele INTEGER NOT NULL,
                    CLoss INTEGER NOT NULL,
                    DetLabels INTEGER NOT NULL,
                    TestData INTEGER NOT NULL,
                    B1T1 INTEGER NOT NULL,
                    B1T0 INTEGER NOT NULL,
                    B0T1 INTEGER NOT NULL,
                    B0T0 INTEGER NOT NULL,
                    B1CF INTEGER NOT NULL,
                    B1CN INTEGER NOT NULL,
                    B0CF INTEGER NOT NULL,
                    B0CN INTEGER NOT NULL,
                    Zero INTEGER NOT NULL,
                    Eins INTEGER NOT NULL,
                    Zwei INTEGER NOT NULL,
                    Drei INTEGER NOT NULL,
                    Vier INTEGER NOT NULL,
                    Fuenf INTEGER NOT NULL,
                    Sechs INTEGER NOT NULL,
                    Sieben INTEGER NOT NULL,
                    Acht INTEGER NOT NULL,
                    Sixers INTEGER NOT NULL,
                    Eighters INTEGER NOT NULL,
                    Empty INTEGER NOT NULL,
                    LabelNum1 INTEGER NOT NULL,
                    LabelNum2 INTEGER NOT NULL,
                    LabelNum3 INTEGER NOT NULL,
                    LabelNum4 INTEGER NOT NULL,
                    LabelNum5 INTEGER NOT NULL,
                    LabelNum6 INTEGER NOT NULL,
                    LabelNum7 INTEGER NOT NULL,
                    LabelNum8 INTEGER NOT NULL,
                    LabelNum9 INTEGER NOT NULL,
                    Precision FLOAT NOT NULL,
                    Recall REAL NOT NULL,
                    FVal REAL NOT NULL,
                    PRIMARY KEY(CutPar, CTN, Dele, CLoss, DetLabels, TestData)
                    );"""
            Cur.execute(sql_command)
            Con.commit()
            
        sql_command = "INSERT INTO TC3_Test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        Cur.executemany(sql_command, results_list)
        Con.commit()
        Con.close()

        print("Finished " + str(Paras))

                

