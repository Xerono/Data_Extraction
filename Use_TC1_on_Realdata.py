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
    LabelDict["[CLS]"] = -100
    LabelDict["[SEP]"] = -100
    LabelDict["Nul"] = 0
    LabelDict["Noise"] = 6
    LabelDict["Grad"] = 1
    LabelDict["Min"] = 2
    LabelDict["Sek"] = 3
    LabelDict["Latitude"] = 4
    LabelDict["Longitude"] = 5
    LabelDict["Padded"] = -100
    IntToLabel = {}
    IntToLabel[0] = "Nul"
    IntToLabel[1] = "Grad"
    IntToLabel[2] = "Min"
    IntToLabel[3] = "Sek"
    IntToLabel[4] = "Latitutde"
    IntToLabel[5] = "Longitude"
    IntToLabel[6] = "Noise"
    IntToLabel[-100] = ["[SEP]","[CLS]", "Padded"]
    return LabelDict, IntToLabel

LabelDict, IntLabelDict = labels_to_int()

Model_Path = ModPath + "TC1_Model_Coordinates_Fake/"



import torch
from transformers import DistilBertForTokenClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = DistilBertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
Maxlength = 917
Model.eval()

import re
def split_string(Par):
    ParSpl = Par.split(" ")
    Splitted = []
    for item in ParSpl:
        Number = False
        for char in item:
            if char.isdigit():
                Number = True
        if Number:
            for char in list(item):
                Splitted.append(char)
        else:
            Splitted.append(item)
    Returnpar = Splitted[0]
    for word in Splitted[1:]:
        Returnpar = Returnpar + " " + word
    return Returnpar

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
        Dataset.append((Coords, 6, split_string(SplitPar)))
        Numbers[0] += 1
for (Coords, Regex, SplitPar) in Eighters:
    if len(SplitPar) < Maxlength:
        Dataset.append((Coords, 8, split_string(SplitPar)))
        Numbers[1] += 1
for SplitPar in NotFound:
    if len(SplitPar) < Maxlength:
        Dataset.append(([], 0, split_string(SplitPar)))
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


def get_token_class(Tokens, Labels):
    Classes = []
    for i in range(len(Tokens)):
        MaxClass = max(Labels[i])
        CurClasses = []
        for j in range(len(Labels[i])):
            if Labels[i][j] == MaxClass:
                CurClasses.append(j)
        Classes.append(CurClasses)
    return Classes

def extract_relevant_classes(Tokens, Classes):
    Relevant = []
    for i in range(len(Tokens)):
        for Element in Classes[i]:
            PotClasses = []
            if Element in [1, 2, 3, 4, 5]:
                PotClasses.append(Element)
        Relevant.append((Tokens[i], PotClasses))
    return Relevant


def Extend(List):
    RList = []
    for item in List:
        for item2 in List:
            RList.append(str(item) + str(item2))
    RList = List + RList
    return RList

def ToCoords(RevTokens):
    rnum = 0
    Grad = []
    Min = []
    Sek = []
    Lat = []
    Long = []
    for (Token, ClassList) in RevTokens:
        for Class in ClassList:
            if Class == 1:
                Grad.append(Token)
            if Class == 2:
                Min.append(Token)
            if Class == 3:
                Sek.append(Token)
            if Class == 4:
                Lat.append(Token)
            if Class == 5:
                Long.append(Token)
    GradE = Extend(Grad)
    MinE = Extend(Min)
    SekE = Extend(Sek)
    LatE = Extend(Lat)
    LongE = Extend(Long)
    DirE = Extend(LatE+LongE)
    if len(SekE)>0:
        rnum = 8
    else:
        rnum = 6
    return(GradE, MinE, SekE, DirE, rnum)

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

for (PotCords, LenCoords, SplitPar) in Dataset:
    Tokens = Tokenizer.tokenize(SplitPar)
    Labels = get_label(SplitPar, Model, Tokenizer)
    Classes = get_token_class(Tokens, Labels)
    RevTokens = extract_relevant_classes(Tokens, Classes)

    if LenCoords == 0:
        if len(RevTokens)>0:
            Resultsdict[10] += 1
        else:
            Resultsdict[00] += 1
    else:
        if len(RevTokens)>0:
            Resultsdict[11] += 1
        else:
            Resultsdict[o1] += 1

        (GradE, MinE, SekE, DirE, rnum) = ToCoords(RevTokens)
        if rnum != LenCoords:
            if len(RevTokens)>=0:
                Resultsdict[B1N] += 1
            else:
                Resultsdict[B0N] += 1
        else:
            ReturnCoords = []
            Hits = 0
            if LenCoords == 8:
                for grad in GradE:
                    if grad == PotCords[0]:
                        Hits += 1
                    if grad == PotCords[4]:
                        Hits += 1
                for mint in MinE:
                    if mint == PotCords[1]:
                        Hits += 1
                    if mint == PotCords[5]:
                        Hits += 1
                for sek in SekE:
                    if sek == PotCords[2]:
                        Hits += 1
                    if sek == PotCords[6]:
                        Hits += 1
                for dire in DirE:
                    if dire == PotCords[3]:
                        Hits += 1
                    if dire == PotCords[7]:
                        Hits += 1
            else:
                for grad in GradE:
                    if grad == PotCords[0]:
                        Hits += 1
                    if grad == PotCords[3]:
                        Hits += 1
                for mint in MinE:
                    if mint == PotCords[1]:
                        Hits += 1
                    if mint == PotCords[4]:
                        Hits += 1
                for dire in DirE:
                    if dire == PotCords[2]:
                        Hits += 1
                    if dire == PotCords[5]:
                        Hits += 1                

            if Hits == LenCoords:
                Resultsdict[B1F] += 1
            else:
                Resultsdict[B1N] += 1
    Runner+=1
    if Runner%1000==0:
        print(str(Runner) + "/" + str(len(Dataset)))


        
results_list = []

ModName = "TC1_Fake_Realdata"

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
    
