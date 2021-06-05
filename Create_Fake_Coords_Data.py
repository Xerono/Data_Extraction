# Load original paragraphs
import os
import sqlite3

CurDir = os.getcwd()
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()

xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

# Find coordinates and beautify

Symbols = ["•", "H", "V", "¢", ".", "j", "J", "°", ",", ";", "Њ", "Ј", "U",
           '"', "″", "'", "o", "@", "؇", "-", "¶", "(", ")", "Љ", "±",
           ":", "µ", "/",
           "8", "9"] # Found by trial & error
Additional_Symbols = ["and"] # used by coordinates, not used in creation
import re


def remove_symbols(potcord): # Obsolete by re.sub
    for symbol in Symbols:
        potcord = potcord.replace(symbol, " ")
    return potcord

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


def convert(potcord):
    potcord = removal(potcord)
    return potcord.split(" ")

# Find xx°xx'N
rege1 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,10}[0-9][0-9].{1,4}(?:W|E).{1,1}"
# Find xx°xx'xx''N
rege2 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,6}[0-9][0-9].{1,4}(?:W|E).{1,1}"
regelist = [rege1, rege2]

Sixers = []
Eighters = []
Errors = []
NotFound = []

cutofflength = 3 # to get less false positives on cost of true positives
for regex in regelist:
    for (FPID, File, Par) in OriginalPars:
        res = re.findall(regex, Par)
        if len(res)>0:
            for potcord in res:
                someerror = False
                pc = convert(potcord)
                for item in pc:
                    if len(item)>=cutofflength:
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
            if len(Par)>30:
                NotFound.append(Par)

import random

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
        
    

def generate_six_coords():
    gradN = str(random.randint(0, 90))
    minN = str(random.randint(0, 59))
    NS = random.choice(["N", "S"])
    gradW = str(random.randint(0, 90))
    minW = str(random.randint(0, 59))
    WE = random.choice(["W", "E"])
    Labels = []
    Coords = []
    if len(gradN) == 1:
        Labels.append("Grad")
        Coords.append(gradN)
    else:
        Labels.append("Grad")
        Labels.append("Grad")
        Coords.append(gradN[0])
        Coords.append(gradN[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    if len(minN) == 1:
        Labels.append("Min")
        Coords.append(minN)
    else:
        Labels.append("Min")
        Labels.append("Min")
        Coords.append(minN[0])
        Coords.append(minN[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    Coords.append(NS)
    Labels.append("Latitude")
    
    
    if len(gradW) == 1:
        Labels.append("Grad")
        Coords.append(gradW)
    else:
        Labels.append("Grad")
        Labels.append("Grad")
        Coords.append(gradW[0])
        Coords.append(gradW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    if len(minW) == 1:
        Labels.append("Min")
        Coords.append(minW)
    else:
        Labels.append("Min")
        Labels.append("Min")
        Coords.append(minW[0])
        Coords.append(minW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    Coords.append(WE)
    Labels.append("Longitude")
    CoordsString = ""
    for i in Coords:
        CoordsString += i
    return (CoordsString, Labels)

def generate_eight_coords():
    gradN = str(random.randint(0, 90))
    minN = str(random.randint(0, 59))
    sekN = str(random.randint(0, 59))
    NS = random.choice(["N", "S"])
    gradW = str(random.randint(0, 90))
    minW = str(random.randint(0, 59))
    sekW = str(random.randint(0, 59))
    WE = random.choice(["W", "E"])
    Labels = []
    Coords = []
    if len(gradN) == 1:
        Labels.append("Grad")
        Coords.append(gradN)
    else:
        Labels.append("Grad")
        Labels.append("Grad")
        Coords.append(gradN[0])
        Coords.append(gradN[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    if len(minN) == 1:
        Labels.append("Min")
        Coords.append(minN)
    else:
        Labels.append("Min")
        Labels.append("Min")
        Coords.append(minN[0])
        Coords.append(minN[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")

    if len(sekN) == 1:
        Labels.append("Sek")
        Coords.append(sekN)
    else:
        Labels.append("Sek")
        Labels.append("Sek")
        Coords.append(sekN[0])
        Coords.append(sekN[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    Coords.append(NS)
    Labels.append("Latitude")
    
    
    if len(gradW) == 1:
        Labels.append("Grad")
        Coords.append(gradW)
    else:
        Labels.append("Grad")
        Labels.append("Grad")
        Coords.append(gradW[0])
        Coords.append(gradW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    if len(minW) == 1:
        Labels.append("Min")
        Coords.append(minW)
    else:
        Labels.append("Min")
        Labels.append("Min")
        Coords.append(minW[0])
        Coords.append(minW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
    
    if len(sekW) == 1:
        Labels.append("Sek")
        Coords.append(sekW)
    else:
        Labels.append("Sek")
        Labels.append("Sek")
        Coords.append(sekW[0])
        Coords.append(sekW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append("Noise")
        
    Coords.append(WE)
    Labels.append("Longitude")
    CoordsString = ""
    for i in Coords:
        CoordsString += i
    return (CoordsString, Labels)

from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)

Maxlength = 917

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Replace(RegexFound, Par, CoordsNew, Labels):
        NewPar = Par.replace(Regexfound, CoordsNew)
        newpar = split_string(NewPar)
        splitted_newpar = Tokenizer.tokenize(newpar)
        Labellist = []
        for i in range(len(splitted_newpar)):
            Labellist.append("Nul")
        CoordsSplit = Tokenizer.tokenize(split_string(CoordsNew))
        for i in range(0, len(splitted_newpar)-len(CoordsSplit)+1):
            if splitted_newpar[i:i+len(CoordsSplit)] == CoordsSplit:
                for j in range(len(Labels)):
                    Labellist[i+j] = Labels[j]
        return newpar, Labellist

NumOfExamples = 100000
Dataset = []
Runner = 1
import time
starttime = time.time()
while Runner <= NumOfExamples:
    if random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])>3:
        if random.choice([1, 2]) == 1:
            CoordsNew, Labels = generate_six_coords()
            random.shuffle(Sixers)
            (Coords, Regexfound, Par) = Sixers[0]
            newpar, Labellist = Replace(Regexfound, Par, CoordsNew, Labels)
        else:
            CoordsNew, Labels = generate_eight_coords()
            random.shuffle(Eighters)
            (Coords, Regexfound, Par) = Eighters[0]
            newpar, Labellist = Replace(Regexfound, Par, CoordsNew, Labels)
    else:
        random.shuffle(NotFound)
        newpar = split_string(NotFound[0])
        splitted_ex = Tokenizer.tokenize(newpar)
        Labellist = []
        for i in range(len(splitted_ex)):
            Labellist.append("Nul")
    if len(newpar)<Maxlength:
        Dataset.append(((Runner, newpar), Labellist))
        Runner+=1
    if Runner % 1000 == 0:
        print(str(Runner) + "/" + str(NumOfExamples))
        print(time.time()-starttime)

import pickle
with open(CurDir + "/Files/FakeData.pickle", 'wb') as handle:
    pickle.dump(Dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

