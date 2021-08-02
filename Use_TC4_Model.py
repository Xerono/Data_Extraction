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
    model = BertForTokenClassification.from_pretrained(PreTrainedModel, num_labels=9).to(device) # delete later
    model.eval()

    LabelDict = {}
    for i in range(1,10):
        LabelDict[i] = 0
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
                        New_Label.append(1)
                    else:
                        New_Label.append(0)
                LabelsForPar.append(New_Label)

            for i in range(len(TokenizedPar)):
                print(TokenizedPar[i])
                print(LabelsForPar[i])
            input()
                    

            
