import os
import pickle

Treshold = float(0.7)


CurDir = os.getcwd()


import sqlite3
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

ResDatabase = CurDir + "/Results/Results.db"

Maxlength = 917

import Module_Coordinates as mc

Dataset = []
Numbers = [0,0,0]
LabelDict, IntLabelDict = mc.labels_to_int()
for (FPID, File, Par) in OriginalPars:
    (Six, Eight, NE, E) = mc.find_coordinates(Par)
    CordsInThis = []
    Numbers[0] = Numbers[0] + len(Six)
    Numbers[1] = Numbers[1] + len(Eight)
    if len(Six) + len(Eight) == 0:
        Numbers[2] = Numbers[2] + 1
    for (Coords, StringC, Par) in Six + Eight:
        CordsInThis.append((Coords, StringC))
    Dataset.append((Par, CordsInThis))

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
    if "TC4" in mdl:
        Models.append(mdl)
Models.append("TC4_1111") # delete later
for mdl in Models:
    Paras = mdl.split("_")[1]

    Cut_Par = bool(int(Paras[0]))
    CTN = bool(int(Paras[1]))
    Dele = bool(int(Paras[2]))
    DLabels = bool(int(Paras[3]))

    Model_Path = ModPath + mdl

    if DLabels:
        num_labels = 12
    else:
        num_labels = 9
    #model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=num_labels).to(device)
    model = BertForTokenClassification.from_pretrained(PreTrainedModel, num_labels=12).to(device) # delete later
    model.eval()


    Zero_Label = []
    for i in range(num_labels):
        Zero_Label.append(float(0))
    
    Basic_Label_Noise = Zero_Label.copy()
    Basic_Label = Zero_Label.copy()
    Basic_Label[2] = float(1)
    Basic_Label_Noise[2] = float(1)
    # Padded, Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
    # Padded, Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long
    # Grad, Min, (Sek), Lat, Grad, Min, (Sek), Long
    All_Results = {}
    for i in range(12):
        All_Results[i] = 0
    AvgPerToken = []
    AvgPerPar = []
    Total_Corrects = 0
    Coord_Correct = 0
    Coord_False = 0
    FalsePositiveCoords = 0
    for (Par, ListOfCoords) in Dataset:
        SplitPar = mc.split_string(Par)
        if len(SplitPar)<Maxlength:
            Full_Labels = []
            TokenizedPar = Tokenizer.tokenize(SplitPar)
            for i in range(len(TokenizedPar)):
                Full_Labels.append(Basic_Label.copy())
            All_Labels = []
            for (PotCords, StringCords) in ListOfCoords:
                i = 0
                TokenizedCooStr = Tokenizer.tokenize(mc.split_string(StringCords))
                TokenizedCoords = []
                for i in range(len(TokenizedPar)-len(TokenizedCooStr)):
                    if TokenizedPar[i:i+len(TokenizedCooStr)] == TokenizedCooStr:
                        StartOfCoords = i
                        
                clabels = []
                for i in range(len(TokenizedCooStr)):
                    clabels.append(Basic_Label_Noise.copy())
                CordL = []
                for k in range(len(PotCords)):
                    Tokenized_KoordAnteil = Tokenizer.tokenize(mc.split_string(PotCords[k]))
                    AnteilLabels = []
                    for j in range(len(Tokenized_KoordAnteil)):
                        CurL = Zero_Label.copy()
                        CurL[3] = float(1)
                        if num_labels == 9:
                            if len(PotCords) == 6:
                                if k == 0 or k == 3:
                                    CurL[4] = float(1)
                                if k == 1 or k == 4:
                                    CurL[5] = float(1)
                                if k == 2:
                                    CurL[7] = float(1)
                                if k == 5:
                                    CurL[8] = float(1)
                            else:
                                if k == 0 or k == 4:
                                    CurL[4] = float(1)
                                if k == 1 or k == 5:
                                    CurL[5] = float(1)
                                if k == 2 or k == 6:
                                    CurL[6] = float(1)
                                if k == 3:
                                    CurL[7] = float(1)
                                if k == 7:
                                    CurL[8] = float(1)
                        else: # 12 Label
                            if len(PotCords) == 6:
                                if k == 0:
                                    CurL[4] = float(1)
                                if k == 1:
                                    CurL[5] = float(1)
                                if k == 2:
                                    CurL[7] = float(1)
                                if k == 3:
                                    CurL[9] = float(1)
                                if k == 4:
                                    CurL[10] = float(1)
                                if k == 5:
                                    CurL[8] = float(1)
                            else:
                                if k == 0:
                                    CurL[4] = float(1)
                                if k == 1:
                                    CurL[5] = float(1)
                                if k == 2:
                                    CurL[6] = float(1)
                                if k == 3:
                                    CurL[7] = float(1)
                                if k == 4:
                                    CurL[9] = float(1)
                                if k == 5:
                                    CurL[10] = float(1)
                                if k == 6:
                                    CurL[11] = float(1)
                                if k == 7:
                                    CurL[8] = float(1)
                        AnteilLabels.append(CurL)
                    CordL.append((AnteilLabels, Tokenized_KoordAnteil))
                for (ccLabels, TKA) in CordL:
                    CFF = False
                    for i in range(len(TokenizedCooStr)):
                        if not CFF and TokenizedCooStr[i:i+len(TKA)] == TKA:
                            for j in range(len(ccLabels)):
                                clabels[i+j] = ccLabels[j]
                            CFF = True
                All_Labels.append((clabels, StartOfCoords))
            
            for (ccLabels, SoC) in All_Labels:
                for i in range(len(ccLabels)):
                    Full_Labels[SoC+i] = ccLabels[i]
                        
            # SplitPar
            # Full_Labels
            StrEnc = Tokenizer(SplitPar, return_tensors="pt").to(device)
            Output = model(**StrEnc)
            Logits = Output.logits[0][1:-1].sigmoid()
            LabelsForPar = []
            for i in range(len(TokenizedPar)):
                Current_Token = TokenizedPar[i]
                Current_Label = Logits[i]
                New_Label = []
                for val in Current_Label:
                    if val.item() > Treshold:
                        New_Label.append(float(1))
                    else:
                        New_Label.append(float(0))
                LabelsForPar.append(New_Label)
                
            Sum_For_Par = 0
            for i in range(len(TokenizedPar)):
                Sum_For_Token = 0
                for j in range(num_labels):
                    if LabelsForPar[i][j] == Full_Labels[i][j]:
                        Total_Corrects += 1
                    All_Results[j] += LabelsForPar[j]
                    Sum_For_Token += LabelsForPar[j]
                Sum_For_Token = Sum_For_Token/num_labels
                Sum_For_Par += Sum_For_Token
                AvgPerToken.append(Sum_For_Token)

                if Full_Labels[i][3] == float(1):
                    if LabelsForPar[i][3] == float(1):
                        Coord_Correct += 1
                    else:
                        Coord_False += 1
                else:
                    if LabelsForPar[i][3] == float(1):
                        FalsePositiveCoords += 1
                
            Sum_For_Par = Sum_For_Par / len(TokenizedPar)
            AvgPerPar.append(Sum_For_Par)
    AvgPerPar = sum(AvgPerPar)/len(AvgPerPar)
    AvgPerToken = sum(AvgPerToken)/len(AvgPerToken)

    results_list = [(int(Cut_Par), int(CTN), int(Dele), int(DLabels), Treshold,
                     Total_Corrects, Coord_Correct, Coord_False, FalsePositiveCoords, AvgPerPar, AvgPerToken,
                     All_Results[0], All_Results[1], All_Results[2], All_Results[3], All_Results[4], All_Results[5], 
                     All_Results[6], All_Results[7], All_Results[8], All_Results[9], All_Results[10], All_Results[11]
                     )]
                     
                    
    Con = sqlite3.connect(ResDatabase)
    Cur = Con.cursor()
    sql_command = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='TC4'"
    res = Cur.execute(sql_command).fetchall()
    if res[0][0] == 0:
        sql_command = """
                CREATE TABLE TC4 (
                CutPar INTEGER NOT NULL,
                CTN INTEGER NOT NULL,
                Dele INTEGER NOT NULL,
                DetLabels INTEGER NOT NULL,
                Treshold FLOAT NOT NULL,
                Total_Corrects INTEGER NOT NULL,
                Coord_Correct INTEGER NOT NULL,
                Coord_False INTEGER NOT NULL,
                AvgPerPar FLOAT NOT NULL,
                AvgPerToken FLOAT NOT NULL,
                Class0 INTEGER NOT NULL,
                Class1 INTEGER NOT NULL,
                Class2 INTEGER NOT NULL,
                Class3 INTEGER NOT NULL,
                Class4 INTEGER NOT NULL,
                Class5 INTEGER NOT NULL,
                Class6 INTEGER NOT NULL,
                Class7 INTEGER NOT NULL,
                Class8 INTEGER NOT NULL,
                Class9 INTEGER NOT NULL,
                Class10 INTEGER NOT NULL,
                Class11 INTEGER NOT NULL,
                PRIMARY KEY(CutPar, CTN, Dele, DetLabels, Treshold)
                );"""
        Cur.execute(sql_command)
        Con.commit()
    sql_command = "INSERT INTO TC4 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    Cur.executemany(sql_command, results_list)
    Con.commit()
    Con.close()
