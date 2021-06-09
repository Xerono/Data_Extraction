import os

CurDir = os.getcwd()

ModPath = CurDir + "/Models/"

import sqlite3
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

def labels_to_int():
    LabelDict = {}
    LabelDict["Nul"] = 0
    LabelDict["[Coordinate"] = 1
    LabelDict["Irrelevant"] = -100
    IntToLabel = {}
    IntToLabel[0] = "Nul"
    IntToLabel[1] = "Coordinate"
    IntToLabel[-100] = ["[SEP]","[CLS]", "Padded"]
    return LabelDict, IntToLabel

LabelDict, IntLabelDict = labels_to_int()

Model_Path = ModPath + "TC2_Model_Coordinates/"



import torch
from transformers import DistilBertForTokenClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = DistilBertForTokenClassification.from_pretrained(Model_Path, num_labels=2).to(device)
from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
Maxlength = 917
Model.eval()

Sixers = []
Eighters = []
Errors = []
NotFound = []
import Module_Coordinates as mc
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
print(Numbers)
        
def get_label(Str, Model, Tokenizer):
    StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
    Output = Model(**StrEnc)
    Softmaxed = Output.logits.softmax(1)
    Labels = []
    for labels in Softmaxed[0]:
        lblcur = []
        for lbl in labels:
            lblcur.append(lbl.item())
        Labels.append(lblcur)
    return Labels[1:-1] # CLS and SEP


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

Runner = 0

def get_classes(Labels):
    Classes = []
    for item in Labels:
        if item[0] > item[1]:
            Classes.append(0)
        else:
            Classes.append(1)
    return Classes

for (PotCords, Regex, Par) in Dataset:
    Tokens = Tokenizer.tokenize(Par)
    Labels = get_label(Par, Model, Tokenizer)

    Classes = get_classes(Labels)
    Found_Coords = []
    for i in range(len(Classes)):
        if Classes[i] != 0:
            Found_Coords.append(Tokens[i])

    if len(PotCords) == 0:
        if len(Found_Coords)>0:
            Resultsdict[10] += 1
        else:
            Resultsdict[o0] += 1
    else:
        if len(Found_Coords)>0:
            Resultsdict[11] += 1
        else:
            Resultsdict[o1] += 1

        TokenCoords = []
        for Coord in PotCords:
            TokenCoords.append(Tokenizer.tokenize(Coord))
        for Tokengroup in TokenCoords:
            Found = True
            for Token in Tokengroup:
                if Token not in Found_Coords:
                    Found = False
            for Token in Found_Coords:
                if Token not in Tokengroup:
                    Found = False
            if not Found:
                if len(Found_Coords)>0:
                    Resultsdict[B1N] += 1
                else:
                    Resultsdict[B0N] += 1
            else:
                if len(Found_Coords)>0:
                    Resultsdict[B1F] += 1
                else:
                    Resultsdict[B1N] += 1
    Runner+=1
    if Runner%1000==0:
        print(str(Runner) + "/" + str(len(Dataset)))


        
results_list = []

ModName = "TC2_Coordinates"

results_list.append((ModName, Resultsdict[11], Resultsdict[10], Resultsdict[o1], Resultsdict[o0]
                     , Resultsdict[B1F], Resultsdict[B1N], Resultsdict[B0F], Resultsdict[B0N], 100)) 

Database = CurDir + "/Results/Results.db"
if not os.path.isfile(Database):
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()
    sql_command = """
            CREATE TABLE Results (
            Model String NOT NULL,
            Bew_1_Tat_1 INTEGER NOT NULL,
            Bew_1_Tat_0 INTEGER NOT NULL,
            Bew_0_Tat_1 INTEGER NOT NULL,
            Bew_0_Tat_0 INTEGER NOT NULL,
            B1_Soil_Found INTEGER NOT NULL,
            B1_Soil_Not_Found INTEGER NOT NULL,
            B0_Soil_Found INTEGER NOT NULL,
            B0_Soil_Not_Found INTEGER NOT NULL,
            Percentage NOT NULL,
            PRIMARY KEY(Model, Percentage)
            );"""
    Cur.execute(sql_command)
    Con.commit()
    Con.close()

Con = sqlite3.connect(Database)
Cur = Con.cursor()
sql_command = "INSERT INTO Results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
Cur.executemany(sql_command, results_list)
Con.commit()
Con.close()
print("Finished")
    
