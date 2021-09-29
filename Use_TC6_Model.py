import os
import pickle
import time
Tresholds = [float(0.999), float(0.9999), float(0.99999), float(0.999999)]
#Tresholds = [float(0.4), float(0.5), float(0.6), float(0.7), float(0.8), float(0.9), float(0.95), float(0.99)]


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
    if "TC6_0" in mdl or "TC6_1" in mdl:
        Models.append(mdl)
for mdl in Models:
    for Treshold in Tresholds:
        Paras = mdl.split("_")[1]

        Cut_Par = bool(int(Paras[0]))
        CTN = bool(int(Paras[1]))
        Dele = bool(int(Paras[2]))
        DLabels = bool(int(Paras[3]))

        Model_Path = ModPath + mdl

        if DLabels:
            num_labels = 11
        else:
            num_labels = 8
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=num_labels).to(device)

        model.eval()

        Zero_Label = []
        for i in range(num_labels):
            Zero_Label.append(float(0))
        
        Basic_Label_Noise = Zero_Label.copy()
        Basic_Label = Zero_Label.copy()
        Basic_Label[0] = float(1)
        Basic_Label_Noise[1] = float(1)
        # Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
        # Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long
        # Grad, Min, (Sek), Lat, Grad, Min, (Sek), Long
        All_Results = {}
        for i in range(11):
            All_Results[i] = 0
        AvgPerToken = []
        AvgPerPar = []
        Total_Corrects = 0
        Coord_Correct = 0
        Coord_False = 0
        FalsePositiveCoords = 0
        Counter = 0
        RealCoordinate = 0
        Num_Of_All_Tokens = 0
        Correct_Labels = 0
        Starttime = time.time()
        for (Par, ListOfCoords) in Dataset:
            if Counter % 10000 == 0:
                print(mdl + " - "  + str(Treshold) + " - " + str(Counter) + "/" + str(len(Dataset)) + " - " +  str(time.time() - Starttime))
            Counter += 1
            SplitPar = Par
            if len(SplitPar)<Maxlength:
                Full_Labels = []
                TokenizedPar = Tokenizer.tokenize(SplitPar)
                for i in range(len(TokenizedPar)):
                    Full_Labels.append(Basic_Label.copy())
                All_Labels = []
                for (PotCords, StringCords) in ListOfCoords: # Find correct labels for each token
                    i = 0
                    TokenizedCooStr = Tokenizer.tokenize(StringCords)
                    TokenizedCoords = []
                    for i in range(len(TokenizedPar)-len(TokenizedCooStr)):
                        if TokenizedPar[i:i+len(TokenizedCooStr)] == TokenizedCooStr:
                            StartOfCoords = i
                            
                    clabels = []
                    for i in range(len(TokenizedCooStr)):
                        clabels.append(Basic_Label_Noise.copy())
                    CordL = []
                    for k in range(len(PotCords)):
                        Tokenized_KoordAnteil = Tokenizer.tokenize(PotCords[k])
                        AnteilLabels = []
                        for j in range(len(Tokenized_KoordAnteil)):
                            CurL = Zero_Label.copy()
                            CurL[2] = float(1)
                            if num_labels == 8:
                                if len(PotCords) == 6:
                                    if k == 0 or k == 3:
                                        CurL[3] = float(1)
                                    if k == 1 or k == 4:
                                        CurL[4] = float(1)
                                    if k == 2:
                                        CurL[6] = float(1)
                                    if k == 5:
                                        CurL[7] = float(1)
                                else:
                                    if k == 0 or k == 4:
                                        CurL[3] = float(1)
                                    if k == 1 or k == 5:
                                        CurL[4] = float(1)
                                    if k == 2 or k == 6:
                                        CurL[5] = float(1)
                                    if k == 3:
                                        CurL[6] = float(1)
                                    if k == 7:
                                        CurL[7] = float(1)
                            else: # 11 Label
                                if len(PotCords) == 6:
                                    if k == 0:
                                        CurL[3] = float(1)
                                    if k == 1:
                                        CurL[4] = float(1)
                                    if k == 2:
                                        CurL[6] = float(1)
                                    if k == 3:
                                        CurL[8] = float(1)
                                    if k == 4:
                                        CurL[9] = float(1)
                                    if k == 5:
                                        CurL[7] = float(1)
                                else:
                                    if k == 0:
                                        CurL[3] = float(1)
                                    if k == 1:
                                        CurL[4] = float(1)
                                    if k == 2:
                                        CurL[5] = float(1)
                                    if k == 3:
                                        CurL[6] = float(1)
                                    if k == 4:
                                        CurL[8] = float(1)
                                    if k == 5:
                                        CurL[9] = float(1)
                                    if k == 6:
                                        CurL[10] = float(1)
                                    if k == 7:
                                        CurL[7] = float(1)
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

                Num_Of_All_Tokens += len(Full_Labels)

                for i in range(len(Full_Labels)):
                    if Full_Labels[i][2] == float(1):
                        RealCoordinate += 1
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
                    if LabelsForPar[i] == Full_Labels[i]:
                        Total_Corrects += 1                    
                    for j in range(num_labels):
                        if LabelsForPar[i][j] == Full_Labels[i][j]:
                            Correct_Labels += 1
                        All_Results[j] += LabelsForPar[i][j]
                        Sum_For_Token += LabelsForPar[i][j]
                    Sum_For_Token = Sum_For_Token/num_labels
                    Sum_For_Par += Sum_For_Token
                    AvgPerToken.append(Sum_For_Token)

                    if Full_Labels[i][2] == float(1):
                        if LabelsForPar[i][2] == float(1):
                            Coord_Correct += 1
                        else:
                            Coord_False += 1
                    else:
                        if LabelsForPar[i][2] == float(1):
                            FalsePositiveCoords += 1
                Sum_For_Par = Sum_For_Par / len(TokenizedPar)
                AvgPerPar.append(Sum_For_Par)


        AvgPerPar = sum(AvgPerPar)/len(AvgPerPar)
        AvgPerToken = sum(AvgPerToken)/len(AvgPerToken)
        Endtime = time.time()
        Final_Time = Endtime - Starttime
        print("Finished " + mdl + " with Treshold " + str(Treshold) + " in " + str(Final_Time))
        KKR = Total_Corrects/Num_Of_All_Tokens
        if Coord_Correct + FalsePositiveCoords != 0:
            Precision = Coord_Correct/(Coord_Correct + FalsePositiveCoords)
        else:
            Precision = 0
        if Coord_Correct + Coord_False != 0:
            Recall = Coord_Correct/(Coord_Correct + Coord_False)
        else:
            Recall = 0
        if Precision + Recall != 0:
            Fval = (2*Precision * Recall)/(Precision + Recall)
        else:
            Fval = 0
        
        results_list = [(int(Cut_Par), int(CTN), int(Dele), int(DLabels), Treshold, Final_Time, Num_Of_All_Tokens,
                         Total_Corrects, Correct_Labels, RealCoordinate, Coord_Correct, Coord_False, FalsePositiveCoords, AvgPerPar, AvgPerToken,
                         All_Results[0], All_Results[1], All_Results[2], All_Results[3], All_Results[4], All_Results[5], 
                         All_Results[6], All_Results[7], All_Results[8], All_Results[9], All_Results[10],
                         KKR, Precision, Recall, Fval
                         )]
                         
                        
        Con = sqlite3.connect(ResDatabase)
        Cur = Con.cursor()
        sql_command = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='TC6'"
        res = Cur.execute(sql_command).fetchall()
        # Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
        if res[0][0] == 0:
            sql_command = """
                    CREATE TABLE TC6 (
                    CutPar INTEGER NOT NULL,
                    CTN INTEGER NOT NULL,
                    Dele INTEGER NOT NULL,
                    DetLabels INTEGER NOT NULL,
                    Treshold FLOAT NOT NULL,
                    Time FLOAT NOT NULL,
                    Num_All_Tokens INTEGER NOT NULL,
                    Coomplete_Correct_Predicted_Tokens INTEGER NOT NULL,
                    Correct_Predicted_Single_Classes INTEGER NOT NULL,
                    Real_Coordinates INTEGER NOT NULL,
                    Correct_Predicted_Coords INTEGER NOT NULL,
                    False_Predicted_Coords INTEGER NOT NULL,
                    FalsePositiveCoords INTEGER NOT NULL,
                    AvgPerPar_AfterTresh FLOAT NOT NULL,
                    AvgPerToken_AfterTresh FLOAT NOT NULL,
                    Class0_Irrelevant INTEGER NOT NULL,
                    Class1_Noise INTEGER NOT NULL,
                    Class2_Coord INTEGER NOT NULL,
                    Class3_Grad1 INTEGER NOT NULL,
                    Class4_Min1 INTEGER NOT NULL,
                    Class5_Sek1 INTEGER NOT NULL,
                    Class6_Lat INTEGER NOT NULL,
                    Class7_Long INTEGER NOT NULL,
                    Class8_Grad2 INTEGER NOT NULL,
                    Class9_Min2 INTEGER NOT NULL,
                    Class10_Sek2 INTEGER NOT NULL,
                    KorrektKlassifikationsRate FLOAT NOT NULL,
                    Precision FLOAT NOT NULL,
                    Recall FLOAT NOT NULL,
                    Fval FLOAT NOT NULL,
                    PRIMARY KEY(CutPar, CTN, Dele, DetLabels, Treshold)
                    );"""
            Cur.execute(sql_command)
            Con.commit()
        sql_command = "INSERT INTO TC6 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        Cur.executemany(sql_command, results_list)
        Con.commit()
        Con.close()
