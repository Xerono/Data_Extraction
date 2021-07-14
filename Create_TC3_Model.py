Basemodel = "distilbert-base-uncased"
Basemodel = "bert-base-cased"


PadLength = 320
DatasetLength = 10000 # Datasetlength / Batch size = Iterations per Epoch
ConvergenceLimit = 0.0001
BackView = 100
Stoptime = 28800 # 8 hours
Batch_Size_Train = 8
Learning_Rate = 5e-5
Custom_Loss = 0.1
TestPercentage = 10


import random
import time
Randomseed = time.time()

Parameters = {}
Parameters["Basemodel"] = Basemodel
Parameters["Randomseed"] = Randomseed
Parameters["PadLength"] = PadLength
Parameters["DatasetLength"] = DatasetLength
Parameters["ConvergenceLimit"] = ConvergenceLimit
Parameters["BackView"] = BackView
Parameters["Stoptime"] = Stoptime
Parameters["Batch_Size_Train"] = Batch_Size_Train
Parameters["Learning_Rate"] = Learning_Rate
Parameters["Custom_Loss"] = Custom_Loss
Parameters["TestPercentage"] = TestPercentage




import os
import sqlite3

CurDir = os.getcwd()
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
MaxLength = 917
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

random.seed(Randomseed)




import Module_Coordinates as mc

PwC = []

for (FPID, File, Par) in OriginalPars:
    (Six, Eight, NF, E) = mc.find_coordinates(Par)
    Found_Coords = Six + Eight
    Coords = []
    if len(Found_Coords)>0 and len(mc.split_string(Par))<MaxLength:
        for (PotCord, StringCord, Par) in Found_Coords:
            Coords.append((PotCord, StringCord))
        PwC.append((Par, Coords))

Alle_Daten = len(PwC)
TestDataLength = int(Alle_Daten/100*TestPercentage)
Testdd = []
Trainingdd = []

for entry in PwC:
    if len(Testdd)< TestDataLength:
        Testdd.append(entry)
    else:
        Trainingdd.append(entry)

import pickle
with open(CurDir + "/Files/TC3_Training.pickle", "wb") as file:
    pickle.dump(Trainingdd, file)
with open(CurDir + "/Files/TC3_Test.pickle", "wb") as file:
    pickle.dump(Testdd, file)
PwC = Trainingdd

LabelDict, IntToLabel = mc.labels_to_int()

import torch


from transformers import BertTokenizerFast
from transformers import BertForTokenClassification





Symbols = ["•", "H", "V", "¢", ".", "j", "J", "°", ",", ";", "Њ", "Ј", "U",
               '"', "″", "'", "o", "@", "؇", "-", "¶", "(", ")", "Љ", "±",
               ":", "µ", "/",
               "8", "9"] # Found by trial & error



def generate_eight_coords(Detailed_Labels):
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
        if Detailed_Labels:
            Labels.append(7)
        else:
            Labels.append(1)
        Coords.append(gradW)
    else:
        if Detailed_Labels:
            Labels.append(7)
            Labels.append(7)
        else:
            Labels.append(1)
            Labels.append(1)
        Coords.append(gradW[0])
        Coords.append(gradW[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(minW) == 1:
        if Detailed_Labels:
            Labels.append(8)
        else:
            Labels.append(2)
        Coords.append(minW)
    else:
        if Detailed_Labels:
            Labels.append(8)
            Labels.append(8)
        else:
            Labels.append(2)
            Labels.append(2)
        
        Coords.append(minW[0])
        Coords.append(minW[1])

    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(sekW) == 1:
        if Detailed_Labels:
            Labels.append(9)
        else:
            Labels.append(3)
        Coords.append(sekW)
    else:
        if Detailed_Labels:
            Labels.append(9)
            Labels.append(9)
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

def generate_six_coords(Detailed_Labels):
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
        if Detailed_Labels:
            Labels.append(7)
        else:
            Labels.append(1)
        Coords.append(gradW)
    else:
        if Detailed_Labels:
            Labels.append(7)
            Labels.append(7)
        else:
            Labels.append(1)
            Labels.append(1)
        Coords.append(gradW[0])
        Coords.append(gradW[1])
        
    for i in range(random.choice([1,2])):
        Coords.append(random.choice(Symbols))
        Labels.append(6)
    
    if len(minW) == 1:
        if Detailed_Labels:
            Labels.append(8)
        else:
            Labels.append(2)
        Coords.append(minW)
    else:
        if Detailed_Labels:
            Labels.append(8)
            Labels.append(8)
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
    (Par, ListOfCoords, Detailed_Labels) = ParCord
    FullNewCoords = []
    for (Coord, String) in ListOfCoords:
        if random.choice([6, 8]) == 6:
            (CoordsString, PotCords, Labels) = generate_six_coords(Detailed_Labels)
        else:
            (CoordsString, PotCords, Labels) = generate_eight_coords(Detailed_Labels)
        Par = Par.replace(String, CoordsString)
        FullNewCoords.append((CoordsString, PotCords, Labels))
        FullLabels = []
        SplitPar = mc.split_string(Par)
        TokenizedSplitPar = Tokenizer.tokenize(SplitPar)
        for i in range(len(TokenizedSplitPar)):
            FullLabels.append(0)
        for (CoordsString, PotCords, Labellist) in FullNewCoords:
            CoordsSplit = Tokenizer.tokenize(mc.split_string(CoordsString))
            for i in range(0, len(TokenizedSplitPar) - len(CoordsSplit) + 1):
                if TokenizedSplitPar[i:i+len(CoordsSplit)] == CoordsSplit:
                    for j in range(len(Labellist)):
                        FullLabels[i+j] = Labellist[j]
        return (SplitPar, FullLabels)






class Dataset(torch.utils.data.Dataset):
    
    
    def __getitem__(self, idx):
        global Storage
        random.shuffle(PwC)
        (Current_Par, CordList) = PwC[0]
        ECoords = []
        SCoords = []
        for (ECord, SCord) in CordList:
            ECoords.append(ECord)
            SCoords.append(SCord)
        if Cut_Par:
            #Paragraph before first coordinates
            FirstCoords = SCoords[0]
            PrePar = Current_Par.split(FirstCoords)[0]
            FirstSplit = PrePar.split(" ")
            NewPre = []
            for Word in FirstSplit:
                if random.choice([1,2,3,4]) != 1:
                    NewPre.append(Word)
            Pre = ""
            for Word in NewPre:
                Pre = Pre + Word + " "
            #Paragraph after last coordinates
            LastCoords = SCoords[-1]
            PostPar = Current_Par.split(LastCoords)[-1]
            LastSplit = PostPar.split(" ")
            NewPost = []
            for Word in LastSplit:
                if random.choice([1,2,3,4]) != 1:
                    NewPost.append(Word)
            Post = ""
            for Word in NewPost:
                Post = Post + Word + " "
            Current_Par = Current_Par.replace(PrePar, Pre)
            Current_Par = Current_Par.replace(PostPar, Post)

        (SP, Labels) = Replace((Current_Par, CordList, Detailed_Labels))

        
        if Coord_To_Noise:
            if not Storage:
                if random.choice([1,2,3,4]) == 1:
                    NewCoords = []
                    NoiseList = []
                    NoisePar = Current_Par
                    for stringcoord in SCoords:
                        coord_list = list(stringcoord)
                        random.shuffle(coord_list)
                        NewCords = ""
                        for item in coord_list:
                            NewCords = NewCords + item
                        NoisePar = NoisePar.replace(stringcoord, NewCords)
                        NoiseList.append(Tokenizer.tokenize(mc.split_string(NewCords)))
                    NoisePar = Current_Par
                       
                    NoisePar = mc.split_string(NoisePar)
                    TNP = Tokenizer.tokenize(NoisePar)
                    FullLabels = []
                    for i in range(len(TNP)):
                        FullLabels.append(0)
                    for Noisel in NoiseList:
                        for i in range(len(TNP)):
                            if i+len(Noisel) < len(TNP) and TNP[i:i+len(Noisel)] == Noisel:
                                for j in range(len(Noisel)):
                                    FullLabels[i+j] = LabelDict["Nul"]
                    Storage = (NoisePar, FullLabels)

                    
            else:
                (SP, Labels) = Storage
                Storage = False
        

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

        if Delete_Teilcoords:
            Slices_With_Coords = []
            for i in range(1, len(Labels)-1):
                if Labels[i] in Interesting_Labels:
                    Slices_With_Coords.append(i)
            if Slices_With_Coords and random.choice([1,2,3,4]) == 1:
                NumberToDelete = random.randint(1, int(len(Slices_With_Coords)))
                Deleted = []
                for i in range(NumberToDelete):
                    ToDelete = random.choice(Slices_With_Coords)
                    while ToDelete in Deleted:
                        ToDelete = random.choice(Slices_With_Coords)
                    Deleted.append(ToDelete)
                for i in Deleted:
                    if i in range(len(TSPcoded)):
                        del TSPcoded[i]
                        del Labels[i]
                        del Attentionmask[i]
                for i in range(len(Labels)):
                    if Labels[i] != LabelDict["Nul"]:
                        Labels[i] = LabelDict["Nul"]
        
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

from transformers import AdamW
from torch.utils.data import DataLoader





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
Options = [False, True]
for Cut_Par in Options:
    for Coord_To_Noise in Options:
        for Delete_Teilcoords in Options:
            for Custom_LossO in Options:
                for Detailed_Labels in Options:
                    random.seed(Randomseed)
                    if Detailed_Labels:
                        Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=11) # Grad1, Min1, Sek1, Lat, Grad2, Min2, Sek2, Long, Noise, Irrelevant, Pad
                    else:
                        Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=8) # Grad, Min, Sek, Lat, Long, Noise, Irrelevant, Pad
                    Tokenizer = BertTokenizerFast.from_pretrained(Basemodel)
                    optim = AdamW(Model.parameters(), lr=Learning_Rate)
                    Training_Loader = DataLoader(TrainData, batch_size = Batch_Size_Train)
                    Model.to(device)
                    loss_history = []
                    convergence = []
                    ConvergenceFound = False
                    Storage = False
                    starttime = time.time()
                    Counter = 0
                    CLoss_History = []
                    Code = str(int(Cut_Par)) + str(int(Coord_To_Noise)) + str(int(Delete_Teilcoords)) + str(int(Custom_LossO)) + str(int(Detailed_Labels))
                    if Detailed_Labels:
                        Interesting_Labels = [1, 2, 3, 4, 5, 7, 8]
                        Missing_Labels = [1, 2, 4, 5, 7, 8]
                    else:
                        Interesting_Labels = [1, 2, 3, 4, 5]
                        Missing_Labels = [1,2,4,5]
                    while not ConvergenceFound and time.time() - starttime < Stoptime :
                        for batch in Training_Loader:
                            optim.zero_grad()
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)
                            outputs = Model(input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs[0]
                            CLossForBatch = 0
                            if Custom_LossO:
                                Softmaxed = outputs.logits.softmax(-1)
                                for SingleParLabels in Softmaxed:
                                    LabelsForPar = []
                                    for ScoresForOneToken in SingleParLabels:
                                        TknMax = 0
                                        for i in range(len(ScoresForOneToken)):
                                            if ScoresForOneToken[i].item() > TknMax:
                                                TknMax = ScoresForOneToken[i].item()
                                        TknLabels = []
                                        for i in range(len(ScoresForOneToken)):
                                            if ScoresForOneToken[i].item() == TknMax:
                                                TknLabels.append(i)
                                        LabelsForPar.append(TknLabels)
                                    WhichCoords = []
                                    for lblistpertoken in LabelsForPar:
                                        lbl = random.choice(lblistpertoken)
                                        if lbl in Interesting_Labels:
                                            WhichCoords.append(lbl)
                                    Add_Loss = 0
                                    if WhichCoords:
                                        # Something is missing (3s are optional)
                                        for Target in Missing_Labels:
                                            if Target not in WhichCoords:
                                                Add_Loss += Custom_Loss
                                        # First Long before Lat
                                        if 4 in WhichCoords and 5 in WhichCoords:
                                            if WhichCoords.index(4) > WhichCoords.index(5):
                                                Add_Loss += Custom_Loss
                                    loss += Add_Loss
                                    CLossForBatch += Add_Loss
                            CLoss_History.append(LossForBatch)
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
                            
                            if Counter % 1000 == 0:
                                print(Code + " with loss of " + str(lossnum))
                            Counter += 1
                               
                    endtime = time.time()
                    FullTime = endtime - starttime
                    
                    mdl = "TC3_" + Code
                    ModName =  mdl + "_Model/"
                    Model.save_pretrained(CurDir + "/Models/" + ModName)
                    
                    
                    HistoryOutputPlace = CurDir + "/Results/TC3_Loss/" + Code + ".pickle"
                    with open(HistoryOutputPlace, "wb") as file:
                        pickle.dump(loss_history, file)
                    if Custom_LossO:
                        CHOP = CurDir + "/Results/TC3_Custom_Loss/" + Code + ".pickle"
                        with open(CHOP, "wb") as file:
                            pickle.dump(CLoss_History, file)

                    Parameters["FullTime"] = FullTime
                    Parameters["Len_los_history"] = len(loss_history)
                        
                    with open(CurDir + "/Models/" + ModName + "/" + "Parameters.pickle", "wb") as file:
                        pickle.dump(Parameters, file)
                    print("Model " + mdl + " saved.")
                        
print("Finished.")
