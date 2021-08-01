import os
import pickle

Debug = False


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

Sixers = []
Eighters = []
Errors = []
NotFound = []
LabelDict, IntLabelDict = mc.labels_to_int()
for (FPID, File, Par) in OriginalPars:
    (Six, Eight, NE, E) = mc.find_coordinates(Par)
    for el in Six:
        Sixers.append(el)
    for el in Eight:
        Eighters.append(el)
    for el in NE:
        NotFound.append(el)
    for el in E:
        Errors.append(el)




Dataset = []
Numbers = [0,0,0]
for (Coords, Regex, SplitPar) in Sixers:
    if len(SplitPar) < Maxlength:
        Dataset.append((Coords, Regex, 6, mc.split_string(SplitPar)))
        Numbers[0] += 1
for (Coords, Regex, SplitPar) in Eighters:
    if len(SplitPar) < Maxlength:
        Dataset.append((Coords, Regex, 8, mc.split_string(SplitPar)))
        Numbers[1] += 1
for SplitPar in NotFound:
    if len(SplitPar) < Maxlength:
        Dataset.append(([], "", 0, mc.split_string(SplitPar)))
        Numbers[2] += 1


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
Models.append("TC4_1111")
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
    Basic_Label = []
    for i in range(num_labels):
        Basic_Label.append(float(0))
    for (PotCords, StringCoords, LenCoords, SplitPar) in Dataset:
        if len(SplitPar)<Maxlength:
            TokenizedPar = Tokenizer.tokenize(SplitPar)
            TokenizedCooStr = Tokenizer.tokenize(StringCoords)
            TokenizedCoords = []
            Basic_Label = []
            i = 0
            for crd in PotCords:
                tokcrd = Tokenizer.tokenize(mc.split_string(crd))
                
                TokenizedCoords.append((tokcrd, i))
                i+=1
            CorrectParLabels = []
            for i in range(len(TokenizedPar)):
                CorrectParLabels.append(Basic_Label.copy())

            
