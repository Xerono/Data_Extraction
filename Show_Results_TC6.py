import os
CurDir = os.getcwd()
Modelpath = CurDir + "/Models/"
print("Which model?")
Paras = input()
print("Treshold?")
Treshold = input()
Treshold = float(Treshold)

if "_" in Paras:
    DLabel = bool(int(Paras.split("_")[0][3]))
else:
    DLabel = bool(int(Paras[3]))

Model_Path = Modelpath + "TC6_" + Paras + "_Model"

from transformers import BertForTokenClassification

PreTrainedModel = 'bert-base-cased'
import torch
import Module_Coordinates as mc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DLabel:
    nlabels = 11
else:
    nlabels = 8
model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=nlabels).to(device)
    
import sqlite3
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()


Sixers = []
Eighters = []
NotFound = []
Errors = []
Maxlength = 917
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
        Dataset.append((Coords, 6, SplitPar))
        Numbers[0] += 1
for (Coords, Regex, SplitPar) in Eighters:
    if len(SplitPar) < Maxlength:
        Dataset.append((Coords, 8, SplitPar))
        Numbers[1] += 1
for SplitPar in NotFound:
    if len(SplitPar) < Maxlength:
        Dataset.append(([], 0, SplitPar))
        Numbers[2] += 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import BertTokenizerFast
if "CT" in Paras:
    Tokenizer = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
else:
    Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)

import random

dict = {}

dict[0] = "Irrelevant"
dict[1] = "Noise"
dict[2] = "Coord"
dict[3] = "Grad1"
dict[4] = "Min1"
dict[5] = "Sek1"
dict[6] = "Lat"
dict[7] = "Long"
dict[8] = "Grad2"
dict[9] = "Min2"
dict[10] = "Sek2"


count = 0
for (PotCords, LenCoords, SplitPar) in Dataset:
    count += 1
    if len(SplitPar)<Maxlength:
        FoundRele = False
        StrEnc = Tokenizer(SplitPar, return_tensors="pt").to(device)
        Tokens = Tokenizer.tokenize(SplitPar)
        maxl = 0
        for word in Tokens:
            if len(word)>maxl:
                maxl = len(word)
        Output = model(**StrEnc)
        Logits = Output.logits[0][1:-1].sigmoid()
        Cur_Labels = []
        Base_Label = []
        for i in range(nlabels):
            Base_Label.append(0)
            
        for vct in Logits:
            CLabel = Base_Label.copy()
            for (i, wert) in enumerate(vct):
                if wert.item() >= Treshold:
                    CLabel[i] = 1
            Cur_Labels.append(CLabel)
        # Irrel, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2

        All_Labels = []
        for (Labelnum, Label) in enumerate(Cur_Labels):
            lbls = []
            for (i, LabelEntry) in enumerate(Label):
                if LabelEntry == 1:
                    lbls.append(dict[i])
            All_Labels.append((Tokens[Labelnum], lbls))
        Cords = False
        for (Tkn, Labls) in All_Labels:
            if 3 in Labls:
                Cords = True
        if Cords or LenCoords:
            print(SplitPar)
            print()
            print(PotCords)
            for (Tkn, Labls) in All_Labels:
                if Labls != [dict[0]]:
                    Tknc = Tkn
                    for k in range(maxl - len(Tknc)):
                        Tknc = Tknc + " "
                    print(Tknc + " " + str(Labls))
            input()
        else:
            print("No coordinates found or labeled in paragraph #" + str(count))
