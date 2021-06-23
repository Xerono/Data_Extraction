import os



modeltype = "a_bc"


CurDir = os.getcwd()

ModPath = CurDir + "/Models/"

Model_Path = ModPath + "TC1" + modeltype + "_Model_Coordinates/"

import transformers
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PreTrainedModel = 'bert-base-cased'
from transformers import BertTokenizerFast
Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
import sqlite3
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()
Maxlength = 917
model.eval()

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

LabelDict, IntToLabel = labels_to_int()

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

def get_label(Str, Model, Tokenizer):
    StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
    Output = Model(**StrEnc)
    Softmaxed = Output.logits.softmax(-1)
    Labels = []
    for labels in Softmaxed[0]:
        lblcur = []
        for lbl in labels:
            lblcur.append(lbl.item())
        Labels.append(lblcur)
    return Labels[1:-1] # CLS and SEP

def extract_relevant_classes(Tokens, Classes):
    Relevant = []
    for i in range(len(Tokens)):
        for Element in Classes[i]:
            PotClasses = []
            if Element in [1, 2, 3, 4, 5]:
                PotClasses.append(Element)
        Relevant.append((Tokens[i], PotClasses))
    return Relevant

def ToCoords(RevTokens):
    rnum = 0
    Grad = []
    Min = []
    Sek = []
    Lat = []
    Long = []
    for (Token, ClassList) in RevTokens:
        if "#" not in Token:
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

for (PotCords, LenCoords, SplitPar) in Dataset:
    Found = False
    if len(SplitPar)<Maxlength:
        Tokens = Tokenizer.tokenize(SplitPar)
        Labels = get_label(SplitPar, model, Tokenizer)
        Classes = get_token_class(Tokens, Labels)
        RevTokens = extract_relevant_classes(Tokens, Classes)

        print(SplitPar)
        print(PotCords)
        for (Token, Label) in RevTokens:
            for lbl in Label:
                print(Token + "   |   " + IntToLabel[lbl])
                Found = True
        if Found:
            input()
