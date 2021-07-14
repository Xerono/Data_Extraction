import re
def find_coordinates(Par):
    Sixers = []
    Eighters = []
    Errors = []
    NotFound = []
    someerror = False
    # Find xx°xx'N
    rege1 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,10}[0-9][0-9].{1,4}(?:W|E)"#.{1,1}"
    # Find xx°xx'xx''N
    rege2 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,6}[0-9][0-9].{1,4}(?:W|E)"#.{1,1}"
    regelist = [rege1, rege2]
    res = []
    
    for regex in regelist:
        res = res + re.findall(regex, Par)
    if len(res)>0:
        for potcord in res:
            pc = convert(potcord)
            for item in pc:
                if len(item)>=3:
                    someerror = True
            if not someerror:
                if len(pc) == 6 and (pc[2] == "N" or pc[2] == "S") and (pc[5] == "W" or pc[5] == "E"):
                    Sixers.append((pc, potcord, Par))
                else:
                    if len(pc) == 8 and (pc[3] == "N" or pc[3] == "S") and (pc[7] == "W" or pc[7] == "E"):
                        Eighters.append((pc, potcord, Par))
                    else:
                        someerror = True
            if someerror:
                Errors.append((pc, potcord, Par))
    else:
        if len(Par)>0:
            NotFound.append(Par)
    return (Sixers, Eighters, NotFound, Errors)

def convert(potcord):
    potcord = removal(potcord)
    return potcord.split(" ")


def remove_middle_twos(potcord):
    reg = "[0-9][0-9]2[0-9][0-9]"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord


def remove_middle_eights(potcord):
    reg = "[0-9][0-9]8[0-9][0-9]"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_end_eights(potcord):
    reg = "[0-9][0-9]8(?:N|S|W|E)"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_middle_nines(potcord):
    reg = "[0-9][0-9]9[0-9][0-9]"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_end_nines(potcord):
    reg = "[0-9][0-9]9(?:N|S|W|E)"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_additional_spaces(potcord):
    while "  " in potcord:
        potcord = potcord.replace("  ", " ")
    if potcord[-1] == " ":
        potcord = potcord[:-1]
    if potcord[0]== " ":
        potcord = potcord[1:]
    return potcord

def remove_additional_w(potcord):
    pc = potcord.split(" ")
    potc = ""
    for part in pc:
        partlist = list(part)
        if len(partlist)==4:
            if partlist[2] == "W":
                partlist[2] = " "
        potco = ""
        for item in partlist:
            potco = potco + item
            
        potc = potc + " " + potco
    return potc

def split_triples(potcord):
    pc = potcord.split(" ")
    returnlist = []
    for part in pc:
        found = False
        partlist = list(part)
        if len(partlist) == 3:
            if partlist[2] in ["N", "S", "W", "E"]:
                p1 = partlist[:2]
                p2 = partlist[2:]
                found = True
        if len(partlist) == 4:
            if partlist[3] in ["N", "S", "W", "E"]:
                p1 = partlist[:2]
                p2 = partlist[3:]
                found = True
        if found:
            pa1 = ""
            for char in p1:
                pa1 = pa1 + char
            pa2 = ""
            for char in p2:
                pa2 = pa2 + char
            returnlist.append(pa1)
            returnlist.append(pa2)
        else:
            returnlist.append(part)
    poco = ""
    for item in returnlist:
        poco = poco + " " + item
    return poco

def removal(potcord):
    potcord = remove_middle_twos(potcord)
    potcord = remove_middle_eights(potcord)
    potcord = remove_end_eights(potcord)
    potcord = remove_middle_nines(potcord)
    potcord = remove_end_nines(potcord)
    potcord = re.sub(r'[^0-9NSWE]', " ", potcord)
    potcord = remove_additional_w(potcord)
    potcord = split_triples(potcord)
    potcord = remove_additional_spaces(potcord)
    return potcord

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
    Returnpar = " "
    for word in Splitted:
        Returnpar = Returnpar + word + " "
    return Returnpar[:-1]

import torch
def get_label(Str, Model, Tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
    Output = Model(**StrEnc)
    Softmaxed = Output.logits.softmax(-1)
    Labels = []
    for labels in Softmaxed[0]:
        lblcur = []
        for lbl in labels:
            lblcur.append(lbl.item())
        Labels.append(lblcur)
    return Labels[1:-1] # CLS + SEP


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

def extract_relevant_classes(Tokens, Classes, ClassList):
    Relevant = []
    for i in range(len(Tokens)):
        for Element in Classes[i]:
            PotClasses = []
            if Element in ClassList:
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
    Grad2 = []
    Min2 = []
    Sek2 = []
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
                if Class == 7:
                    Grad2.append(Token)
                if Class == 8:
                    Min2.append(Token)
                if Class == 9:
                    Sek2.append(Token)
    GradE = Extend(Grad)
    MinE = Extend(Min)
    SekE = Extend(Sek)
    LatE = Extend(Lat)
    Grad2E = Extend(Grad2)
    Min2E = Extend(Min2)
    Sek2E = Extend(Sek2)
    LongE = Extend(Long)
    DirE = Extend(LatE+LongE)
    if len(SekE + Sek2E)>0:
        rnum = 8
    else:
        rnum = 6
    return(GradE, MinE, SekE, DirE, Grad2E, Min2E, Sek2E, rnum)

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
    IntToLabel[4] = "Latitude"
    IntToLabel[5] = "Longitude"
    IntToLabel[6] = "Noise"
    IntToLabel[-100] = ["[SEP]","[CLS]", "Padded"]
    return LabelDict, IntToLabel
