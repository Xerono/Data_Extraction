

Basemodel = "bert-base-cased"
#Basemodel = "distilbert-base-uncased"
Randomseed = 613513
PadLength = 320
MaxLength = 900
DatasetLength = 10000 # Datasetlength / Batch size = Iterations per Epoch
ConvergenceLimit = 0.0001
BackView = 100
Stoptime = 28800 # 8 hours
Batch_Size_Train = 8


import random

random.seed(Randomseed)

import os
import sqlite3

CurDir = os.getcwd()
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
Maxlength = 917
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()


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
    Returnpar = ""
    for word in Splitted:
        Returnpar = Returnpar + word + " "
    return Returnpar[:-1]



import Module_Coordinates as mc

PwC = []


for (FPID, File, Par) in OriginalPars:
    (Six, Eight, NF, E) = mc.find_coordinates(Par)
    Found_Coords = Six + Eight
    Coords = []
    if len(Found_Coords)>0 and len(split_string(Par))<MaxLength:
        for (PotCord, StringCord, Par) in Found_Coords:
            Coords.append((PotCord, StringCord))
        PwC.append((Par, Coords))




import torch


from transformers import BertTokenizerFast
from transformers import BertForTokenClassification



Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=8)
Tokenizer = BertTokenizerFast.from_pretrained(Basemodel)

Symbols = ["•", "H", "V", "¢", ".", "j", "J", "°", ",", ";", "Њ", "Ј", "U",
               '"', "″", "'", "o", "@", "؇", "-", "¶", "(", ")", "Љ", "±",
               ":", "µ", "/",
               "8", "9"] # Found by trial & error



def generate_eight_coords():
    gradN = str(random.randint(0, 90))
    minN = str(random.randint(0, 59))
    sekN = str(random.randint(0, 59))
    NS = random.choice(["N", "S"])
    gradW = str(random.randint(0, 90))
    minW = str(random.randint(0, 59))
    sekW = str(random.randint(0, 59))
    WE = random.choice(["W", "E"])
    PotCoords = (gradN, minN, sekN, NS, gradW, minW, sekW, WE)
    Labels = []
    Coords = []
    if len(gradN) == 1:
        Labels.append(1)
        Coords.append(gradN)
    else:
        Labels.append(1)
        Labels.append(1)
        Coords.append(gradN[0])
        Coords.append(gradN[1])

        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(minN) == 1:
        Labels.append(2)
        Coords.append(minN)
    else:
        Labels.append(2)
        Labels.append(2)
        Coords.append(minN[0])
        Coords.append(minN[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)

    if len(sekN) == 1:
        Labels.append(3)
        Coords.append(sekN)
    else:
        Labels.append(3)
        Labels.append(3)
        Coords.append(sekN[0])
        Coords.append(sekN[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    Coords.append(NS)
    Labels.append(4)

    
    if len(gradW) == 1:
        Labels.append(1)
        Coords.append(gradW)
    else:
        Labels.append(1)
        Labels.append(1)
        Coords.append(gradW[0])
        Coords.append(gradW[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(minW) == 1:
        Labels.append(2)
        Coords.append(minW)
    else:
        Labels.append(2)
        Labels.append(2)
        Coords.append(minW[0])
        Coords.append(minW[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(sekW) == 1:
        Labels.append(3)
        Coords.append(sekW)
    else:
        Labels.append(3)
        Labels.append(3)
        Coords.append(sekW[0])
        Coords.append(sekW[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
        
    Coords.append(WE)
    Labels.append(5)
    CoordsString = ""

    for i in Coords:
        CoordsString += i
    return (CoordsString, PotCoords, Labels)

def generate_six_coords():
    gradN = str(random.randint(0, 90))
    minN = str(random.randint(0, 59))
    NS = random.choice(["N", "S"])
    gradW = str(random.randint(0, 90))
    minW = str(random.randint(0, 59))
    WE = random.choice(["W", "E"])
    PotCoords = (gradN, minN, NS, gradW, minW, WE)
    Labels = []
    Coords = []
    if len(gradN) == 1:
        Labels.append(1)
        Coords.append(gradN)
    else:
        Labels.append(1)
        Labels.append(1)
        Coords.append(gradN[0])
        Coords.append(gradN[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(minN) == 1:
        Labels.append(2)
        Coords.append(minN)
    else:
        Labels.append(2)
        Labels.append(2)
        Coords.append(minN[0])
        Coords.append(minN[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    Coords.append(NS)
    Labels.append(4)
    
    
    if len(gradW) == 1:
        Labels.append(1)
        Coords.append(gradW)
    else:
        Labels.append(1)
        Labels.append(1)
        Coords.append(gradW[0])
        Coords.append(gradW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(minW) == 1:
        Labels.append(2)
        Coords.append(minW)
    else:
        Labels.append(2)
        Labels.append(2)
        Coords.append(minW[0])
        Coords.append(minW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    Coords.append(WE)
    Labels.append(5)
    CoordsString = ""
    for i in Coords:
        CoordsString += i
    return (CoordsString, PotCoords, Labels)









def Replace(ParCord):
    (Par, ListOfCoords) = ParCord
    FullNewCoords = []
    for (Coord, String) in ListOfCoords:
        if random.choice([6, 8]) == 6:
            (CoordsString, PotCords, Labels) = generate_six_coords()
        else:
            (CoordsString, PotCords, Labels) = generate_eight_coords()
        Par = Par.replace(String, CoordsString)
        FullNewCoords.append((CoordsString, PotCords, Labels))
        FullLabels = []
        SplitPar = split_string(Par)
        TokenizedSplitPar = Tokenizer.tokenize(SplitPar)
        for i in range(len(TokenizedSplitPar)):
            FullLabels.append(0)
        for (CoordsString, PotCords, Labellist) in FullNewCoords:
            CoordsSplit = Tokenizer.tokenize(split_string(CoordsString))
            for i in range(0, len(TokenizedSplitPar) - len(CoordsSplit) + 1):
                if TokenizedSplitPar[i:i+len(CoordsSplit)] == CoordsSplit:
                    for j in range(len(Labellist)):
                        FullLabels[i+j] = Labellist[j]
        return (SplitPar, FullLabels)











from transformers import Trainer, TrainingArguments


class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        random.shuffle(PwC)
        (SP, Labels) = Replace(PwC[0])
        TSP = Tokenizer(SP)
        Attentionmask = []
        for i in range(len(Labels)):
            if Labels[i] == 0:
                if random.choice([1,2,3]) == 1:
                    Attentionmask.append(0)
                else:
                    Attentionmask.append(1)
            else:
                Attentionmask.append(1)
        
        TSPcoded = TSP['input_ids'][1:-1]# CLS / SEP
        
        for i in range(PadLength-len(Labels)):
            TSPcoded.append(0)
            Labels.append(-100)
            Attentionmask.append(0)

        item = {}
        item['input_ids'] =  torch.tensor(TSPcoded)
        item['labels'] = torch.tensor(Labels)
        item['attention_mask'] = torch.tensor(Attentionmask)       
        return item

    def __len__(self):
        return DatasetLength

TrainData = Dataset()
TestData = Dataset()


from transformers import AdamW
from torch.utils.data import DataLoader
optim = AdamW(Model.parameters(), lr=5e-5)
Training_Loader = DataLoader(TrainData, batch_size = Batch_Size_Train)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model.to(device)

loss_history = []
Diff_History = []
convergence = []

ConvergenceFound = False

import time

starttime = time.time()
while not ConvergenceFound and time.time() - starttime < Stoptime :
    for batch in Training_Loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = Model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        lossnum = loss.item()
        loss_history.append(lossnum)
        convergence.append(lossnum)
        if len(convergence) == BackView + 1:
            avg = 0
            for i in range(BackView):
                avg += convergence[i]
            LastLoss = avg/BackView
            Diff = LastLoss - convergence[BackView]
            if abs(Diff) < ConvergenceLimit:
                ConvergenceFound = True
            convergence = convergence[1:]
            Diff_History.append(Diff)
            
endtime = time.time()
print(len(loss_history))
FullTime = endtime - starttime
print(FullTime)
ModName = "TC1e_bc_Model_Coordinates/"
Model.save_pretrained(CurDir + "/Models/" + ModName)
print("Model saved.")
import pickle
HistoryOutputPlace = CurDir + "/Results/" + "TC1e_History_" + str(FullTime) + ".pickle"
with open(HistoryOutputPlace, "wb") as file:
    pickle.dump(loss_history, file)
print("Finished.")
