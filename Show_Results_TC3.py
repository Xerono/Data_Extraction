import os
CurDir = os.getcwd()
Modelpath = CurDir + "/Models/"
print("Which model?")
Paras = input()
DLabel = bool(int(Paras[4]))

Model_Path = Modelpath + "TC3_" + Paras + "_Model"

from transformers import BertForTokenClassification

PreTrainedModel = 'bert-base-cased'
import torch
import Module_Coordinates as mc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DLabel:
    model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=11).to(device)
    Rel_Labels = [1,2,3,4,5,7,8,9]
else:
    model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
    Rel_Labels = [1,2,3,4,5]
    
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
        Dataset.append((Coords, 6, mc.split_string(SplitPar)))
        Numbers[0] += 1
for (Coords, Regex, SplitPar) in Eighters:
    if len(SplitPar) < Maxlength:
        Dataset.append((Coords, 8, mc.split_string(SplitPar)))
        Numbers[1] += 1
for SplitPar in NotFound:
    if len(SplitPar) < Maxlength:
        Dataset.append(([], 0, mc.split_string(SplitPar)))
        Numbers[2] += 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import BertTokenizerFast
Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)

import random
random.shuffle(Dataset)


for (PotCords, LenCoords, SplitPar) in Dataset:
    if len(SplitPar)<Maxlength:
        FoundRele = False
        Tokens = Tokenizer.tokenize(SplitPar)
        Labels = mc.get_label(SplitPar, model, Tokenizer)
        Classes = mc.get_token_class(Tokens, Labels)
        RevTokens = mc.extract_relevant_classes(Tokens, Classes, Rel_Labels)
        for (Token, List) in RevTokens:
            for item in List:
                FoundRele = True
        if FoundRele:
            print("Paragraph:")
            print(SplitPar)
            print()
            print("Coordinates to find:")
            print(PotCords)
            print()
            print("Token | Label:")
            for (Token, Labellist) in RevTokens:
                for lbl in Labellist:
                    print(Token + " | " + str(lbl))
            print("------------------------------------------")
            print()
            input()
