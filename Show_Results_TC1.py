import os

RealData = False
OnlyShowFoundTokens = True


modeltype = "a_bc"
modeltype = "a_dc"
modeltype = "a_bu"
#modeltype = "a_du"

#modeltype = "b_bc"
#modeltype = "b_dc"
#modeltype = "b_bu"
#modeltype = "b_du"

#modeltype = "r_bc"
#modeltype = "r_dc"
#modeltype = "r_bu"
#modeltype = "r_du"

#modeltype = "t_bc"
#modeltype = "t_dc"

#modeltype = "CLS"

    
import transformers
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from transformers import BertTokenizerFast
from transformers import DistilBertTokenizerFast

from transformers import BertForTokenClassification
from transformers import DistilBertForTokenClassification

import os

CurDir = os.getcwd()

ModPath = CurDir + "/Models/"

Model_Path = ModPath + "TC1" + modeltype + "_Model_Coordinates/"

if modeltype == "CLS" or modeltype.split("_")[1] == "bc":
    PreTrainedModel = 'bert-base-cased'
    Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
    if modeltype == "CLS":
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=9).to(device)
    else:
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
else:
    if modeltype.split("_")[1] == "dc":
        PreTrainedModel = 'distilbert-base-cased'
        model = DistilBertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
        
    if modeltype.split("_")[1] == "bu":
        PreTrainedModel = 'bert-base-uncased'
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
        Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)

    if modeltype.split("_")[1] == "du":
        PreTrainedModel = 'distilbert-base-uncased'
        model = DistilBertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)



if modeltype == "CLS":
    fdfile = "b_bc"
else:
    if modeltype.split("_")[0] == "r":
        fdfile = False
    else:
        fdfile = modeltype

model.eval()

Maxlength = 917

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

if RealData:
    import sqlite3
    Database = CurDir + "/Files/Database.db"
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()
    xs = "Select * FROM Pars"
    OriginalPars = Cur.execute(xs).fetchall()
    Con.close()
    
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
    Runner = 0
    for (PotCords, LenCoords, SplitPar) in Dataset:
        Runner+=1
        if Runner%1000 == 0:
            print(str(Runner) + "/" + str(len(Dataset)))
        if len(SplitPar)<Maxlength:
            Tokens = Tokenizer.tokenize(SplitPar)
            Labels = get_label(SplitPar, model, Tokenizer)
            Classes = get_token_class(Tokens, Labels)
            RevTokens = extract_relevant_classes(Tokens, Classes)
            Found = False
            wrds = []
            for (Token, Label) in RevTokens:
                for lbl in Label:
                    wrds.append(Token + "   |   " + IntToLabel[lbl])
                    Found = True
            if OnlyShowFoundTokens:
                if Found:
                    print(SplitPar)
                    print(PotCords)
                    for wrd in wrds:
                        print(wrd)
                    input()
            else:
                print(SplitPar)
                print(PotCords)
                if Found:
                    for wrd in wrds:
                        print(wrd)
                    input()
else:
    if not fdfile:
        print("Real models don't have fake data")
    else:
        import pickle
        FakeDataFile = open(os.getcwd() + "/Files/FakeData_" + fdfile + ".pickle", "rb")
        fdata = pickle.load(FakeDataFile)
        Numbers = fdata[0]
        Dataset = fdata[1:]
        Runner = 0
        for ((ID, SplitPar), PotCords, Labellist) in Dataset:
            Runner+=1
            if Runner%1000 == 0:
                print(str(Runner) + "/" + str(len(Dataset)))
            if len(SplitPar)<Maxlength:
                Tokens = Tokenizer.tokenize(SplitPar)
                Labels = get_label(SplitPar, model, Tokenizer)
                Classes = get_token_class(Tokens, Labels)
                RevTokens = extract_relevant_classes(Tokens, Classes)
                Found = False
                wrds = []
                for (Token, Label) in RevTokens:
                    for lbl in Label:
                        wrds.append(Token + "   |   " + IntToLabel[lbl])
                        Found = True
                if OnlyShowFoundTokens:
                    if Found:
                        print(SplitPar)
                        print(PotCords)
                        for wrd in wrds:
                            print(wrd)
                        input()
                else:
                    print(SplitPar)
                    print(PotCords)
                    if Found:
                        for wrd in wrds:
                            print(wrd)
                        input()
