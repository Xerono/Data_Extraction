import os

CurDir = os.getcwd()
Filepath =  CurDir + "/Files/"

Cropsfile = Filepath + "Cropslist.txt"
Texturefile = Filepath + "Texturelist.txt"
Soilfile = Filepath + "Soillist.txt"

Cropslist = []
with open(Cropsfile, "r") as file:
    for line in file.readlines():
        Cropslist.append(line.rstrip())
Texturelist = []
with open(Texturefile, "r") as file:
    for line in file.readlines():
        Texturelist.append(line.rstrip())
Soillist = []
with open(Soilfile, "r") as file:
    for line in file.readlines():
        Soillist.append(line.rstrip())   

import sqlite3
Database = Filepath + "Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

# ID, Filename, Par
import Module_Coordinates as mc
# Target: Par, ListOfSoilsInPar, ListOfTexturesInPar, ListOfCropsInPar, ListOfCoordsInPar
Data = []
import pickle
if not os.path.isfile(CurDir + "/Files/TCF_A_All.pickle"):
    for Count, (ID, Filename, Par) in enumerate(OriginalPars):
        if Count%10000==0:
            print(str(Count) + "/" + str(len(OriginalPars)))
        
        ParCrops = []
        ParTexts = []
        ParSoils = []
        ParCords = []
        for crop in Cropslist:
            crop1 = " " + crop + " "
            crop2 = "(" + crop + " "
            crop3 = " " + crop + ")"
            crop4 = "(" + crop + " "
            crop5 = " " + crop + ","
            edcrops = [crop1, crop2, crop3, crop4, crop5]
            for cropp in edcrops:
                if cropp in Par:
                    ParCrops.append(crop)
        for text in Texturelist:
            text1 = " " + text + " "
            text2 = "(" + text + " "
            text3 = " " + text + ")"
            text4 = "(" + text + " "
            text5 = " " + text + ","
            edtext = [text1, text2, text3, text4, text5]
            for textt in edtext:
                if textt in Par:
                    ParTexts.append(text)
        for soil in Soillist:
            if soil in Par:
                ParSoils.append(soil)
        (Six, Eight, NF, E) = mc.find_coordinates(Par)
        Found_Coords = Six + Eight    
        for (PotCord, StringCord, Par) in Found_Coords:
            ParCords.append((StringCord, PotCord))
        Data.append((Par, (list(set(ParCrops)), list(set(ParTexts)), list(set(ParSoils)), ParCords)))
else:
    with open(CurDir + "/Files/TCF_A_All.pickle", "rb") as file:
        DataLoaded = pickle.load(file)
        for (Par, Labels, (PC, PT, PS, PC2)) in DataLoaded:
             Data.append((Par, (PC, PT, PS, PC2)))


from transformers import BertTokenizerFast



num_labels = 4 # Crop, Soil, Texture, Coordinate
Basemodel = "bert-base-cased"

Tokenizer = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")

No_Class = [float(0), float(0), float(0), float(0)]
Crops_Class = No_Class.copy()
Crops_Class[0] = float(1)
Texture_Class = No_Class.copy()
Texture_Class[1] = float(1)
Soils_Class = No_Class.copy()
Soils_Class[2] = float(1)
Coords_Class = No_Class.copy()
Coords_Class[3] = float(1)


FittingData = []
for (Par, (ParCrops, ParTextures, ParSoils, ParCords)) in Data:
    TokenizedPar = Tokenizer.tokenize(Par)
    if len(TokenizedPar) < 490:
        Labels = []
        for Token in TokenizedPar:
            Labels.append(No_Class)
        for crop in ParCrops:
            TokenCrop = Tokenizer.tokenize(crop)
            for i in range(len(TokenizedPar)-len(TokenCrop)):
                if TokenizedPar[i:i+len(TokenCrop)] == TokenCrop:
                    for j in range(len(TokenCrop)):
                        Labels[i+j] = Crops_Class
        for texture in ParTextures:
            TokenTexture = Tokenizer.tokenize(texture)
            for i in range(len(TokenizedPar)-len(TokenTexture)):
                if TokenizedPar[i:i+len(TokenTexture)] == TokenTexture:
                    for j in range(len(TokenTexture)):
                        Labels[i+j] = Texture_Class
        for soil in ParSoils:
            TokenSoil = Tokenizer.tokenize(soil)
            for i in range(len(TokenizedPar)-len(TokenSoil)):
                if TokenizedPar[i:i+len(TokenSoil)] == TokenSoil:
                    for j in range(len(TokenSoil)):
                        Labels[i+j] = Soils_Class
        for (cord, Extract_Cord) in ParCords:
            TokenCord = Tokenizer.tokenize(cord)
            for i in range(len(TokenizedPar)-len(TokenCord)):
                if TokenizedPar[i:i+len(TokenCord)] == TokenCord:
                    for j in range(len(TokenCord)):
                        Labels[i+j] = Coords_Class
        FittingData.append((Par, Labels, (ParCrops, ParTextures, ParSoils, ParCords)))
print("Created Labels")
Data_With_Things = []
Data_Without_Things = []
for (Par, Labels, (PC, PT, PS, PC2)) in FittingData:
    if PC or PT or PS or PC2:
        Data_With_Things.append((Par, Labels, (PC, PT, PS, PC2)))
    else:
        Data_Without_Things.append((Par, Labels, (PC, PT, PS, PC2)))

TestPercentage = 90
Randomseed = "Final_TC_Alpha"

LenTraining = int(len(Data_With_Things)/100*TestPercentage)
import random
random.seed(Randomseed)
TrainingData = Data_With_Things[:LenTraining]
TestData = Data_With_Things[LenTraining:]

with open(CurDir + "/Files/TCF_A_Training.pickle", "wb") as file:
    pickle.dump(TrainingData, file)
with open(CurDir + "/Files/TCF_A_Test.pickle", "wb") as file:
    pickle.dump(TestData, file)
with open(CurDir + "/Files/TCF_A_All.pickle", "wb") as file:
    pickle.dump(FittingData, file)

    
from transformers import AdamW
from transformers import BertForTokenClassification
from torch.utils.data import DataLoader
import torch
Learning_Rate = 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=num_labels).to(device)
optim = AdamW(Model.parameters(), lr=Learning_Rate)
PadLength = 510
DatasetLength = 10000 # 10000
class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        random.shuffle(TrainingData)
        Par, Labels_DS, (ParCrops, ParTextures, ParSoils, ParCords) = TrainingData[0]
        TokenizedPar_WithCLSandSEP = Tokenizer(Par)
        TokenizedPar = TokenizedPar_WithCLSandSEP['input_ids'][1:-1]
        Attention_Mask = []
        for i in range(len(Labels_DS)):
            if Labels_DS[i] == No_Class and random.choice([1,2,3,4]) == 1:
                    Attention_Mask.append(0)
            else:
                Attention_Mask.append(1)
        for i in range(PadLength - len(TokenizedPar)):
            TokenizedPar.append(0)
            Labels_DS.append(No_Class)
            Attention_Mask.append(0)

        item = {}
        item['input_ids'] = torch.tensor(TokenizedPar)
        item['labels'] = torch.tensor(Labels_DS)
        item['attention_mask'] = torch.tensor(Attention_Mask)
        return item
    def __len__(self):
        return DatasetLength

import time

Batch_Size = 8
Pos_Weight_Vector = torch.ones([num_labels])
Training_Loader = DataLoader(Dataset(), batch_size = Batch_Size)
BCEWLL = torch.nn.BCEWithLogitsLoss(pos_weight = Pos_Weight_Vector).to(device)
Loss_History = []
Counter = 0
Time_For_Batch = []
Custom_Loss = 0
Stoptime = 60 # 28800
Starttime = time.time()

while time.time()-Starttime < Stoptime:
    for batch in Training_Loader:
        Batchstarttime = time.time()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        att_mask = batch['attention_mask'].to(device)
        Output = Model(input_ids, attention_mask = att_mask)
        Logits = Output.logits
        Loss = BCEWLL(Logits, labels)
        lossnum = Loss.item()
        Loss_History.append(lossnum)
        Loss.backward()
        optim.step()
        if Counter % 1000 == 0:
            print("Loss of " + str(round(lossnum, 6)) + " after " + str(Counter) + " batches, " + str(round(time.time() - Starttime, 2)) + "/" + str(Stoptime) + " seconds)")
        Counter += 1
        Batchendtime = time.time()
        Time_For_Batch.append(Batchendtime - Batchstarttime)
Endtime = time.time()
print("Finished Model")
Fulltime = Endtime - Starttime
ModName = "Alpha"
Model.eval()
Model.save_pretrained(CurDir + "/Models/" + ModName)
HistoryOutputPlace = CurDir + "/Results/TCF_Loss/"
if not os.path.isdir(HistoryOutputPlace):
    os.mkdir(HistoryOutputPlace)
with open(HistoryOutputPlace + "Alpha_.pickle", "wb") as file:
    pickle.dump(Loss_History, file)
Parameters = {}
Parameters["FullTime"] = Fulltime
Parameters["Pos_Weight_Vector"] = Pos_Weight_Vector
Parameters["Len_los_history"] = len(Loss_History)
Parameters["Time_Per_Batch"] = Time_For_Batch
Parameters["Basemodel"] = Basemodel
Parameters["Randomseed"] = Randomseed
Parameters["PadLength"] = PadLength
Parameters["DatasetLength"] = DatasetLength
Parameters["Stoptime"] = Stoptime
Parameters["Batch_Size_Train"] = Batch_Size
Parameters["Learning_Rate"] = Learning_Rate
Parameters["Custom_Loss"] = Custom_Loss
Parameters["TestPercentage"] = TestPercentage
with open(CurDir + "/Models/" + ModName + "/" + "Parameters.pickle", "wb") as file:
    pickle.dump(Parameters, file)
print("Model " + ModName + " saved.")

Tresholds = [float(0.4), float(0.5), float(0.6), float(0.7), float(0.8), float(0.9), float(0.95), float(0.99), float(0.999), float(0.9999), float(0.99999), float(0.999999)]
Daten = [(FittingData, "All"), (TrainingData, "Train"), (TestData, "Test")]


ResDatabase = CurDir + "/Results/Results.db"

for (Dater, DescriptorData) in Daten:
    for Treshold in Tresholds:
        print("Starting " + DescriptorData + " with treshold " + str(Treshold))
        Starttime = time.time()
        Full_Correct_Relevant_Labels = 0
        Full_Correct_Labels = 0
        False_Positive_Classes = 0
        Correct_Classes = 0
        False_Classes = 0
        All_Classes = 0
        All_Labels = 0
        Not_Full_Correct_Relevants = 0
        OneLabels = {}
        RealOneLabels = {}
        for i in range(num_labels):
            OneLabels[i] = 0
            RealOneLabels[i] = 0
        for Par_Aus, Labels_Aus, (ParCrops, ParTextures, ParSoils, ParCords) in Dater:
            print(Par)
            print(len(Tokenizer.tokenize(Par)))
            print(len(Labels))
            Calc_Labels_Par = []
            All_Labels += len(Labels_Aus)
            All_Classes += num_labels*len(Labels_Aus)
            TokenizedPar = Tokenizer.tokenize(Par_Aus)
            StrEnc = Tokenizer(Par, return_tensors="pt").to(device)
            Output = Model(**StrEnc)
            Logits = Output.logits[0][1:-1].sigmoid()
            for i in range(len(Labels_Aus)):
                for j in range(num_labels):
                    if Labels[i][j] == float(1):
                        RealOneLabels[j] += 1
            for i in range(len(Labels)):
                Current_Token = TokenizedPar[i]
                Current_Logits = Logits[i]
                Calc_Label = []
                for value in Current_Logits:
                    if value.item() > Treshold:
                        Calc_Label.append(float(1))
                    else:
                        Calc_Label.append(float(0))
                Calc_Labels_Par.append(Calc_Label)
            for i in range(len(Labels_Aus)):
                if Calc_Labels_Par[i] == Labels_Aus[i]:
                    Full_Correct_Labels += 1
                if float(1) in Labels_Aus[i]:
                    if Calc_Labels_Par[i] == Labels_Aus[i]:
                        Full_Correct_Relevant_Labels += 1
                    else:
                        Not_Full_Correct_Relevants += 1
                for j in range(num_labels):
                    if Calc_Labels_Par[i][j] == Labels_Aus[i][j]:
                        Correct_Classes += 1
                    else:
                        False_Classes += 1
                    if Calc_Labels_Par[i][j] == float(1):
                        OneLabels[j] += 1
                    if Labels[i][j] == float(1):
                        if Calc_Labels_Par[i][j] == float(1):
                            False_Positive_Classes += 1
        Endtime = time.time()
        FullTime = Endtime - Starttime
        KKR = Full_Correct_Labels/All_Labels
        KKR_Crops = OneLabels[0]/RealOneLabels[0]
        KKR_Text = OneLabels[1]/RealOneLabels[1]
        KKR_Soil = OneLabels[2]/RealOneLabels[2]
        KKR_Coord = OneLabels[3]/RealOneLabels[3]
        if Correct_Classes + False_Positive_Classes != 0:
            Precision_Classes = Correct_Classes/(Correct_Classes + False_Positive_Classes)
        else:
            Precision_Classes = 0
        if Correct_Classes + False_Classes != 0:
            Recall_Classes = Correct_Classes/(Correct_Classes + False_Classes)
        else:
            Recall_Classes = 0
        if Precision_Classes + Recall_Classes != 0:
            FVal = (2*Precision_Classes * Recall_Classes)/(Precision_Classes + Recall_Classes)
        else:
            FVal = 0
        results_List = [(ModName, Treshold, DescriptorData,
                         FullTime, All_Labels, Full_Correct_Labels, Full_Correct_Relevant_Labels, Not_Full_Correct_Relevants,
                         All_Classes, Correct_Classes, False_Classes,
                         OneLabels[0], OneLabels[1], OneLabels[2], OneLabels[3], RealOneLabels[0], RealOneLabels[1], RealOneLabels[2], RealOneLabels[3],
                         KKR_Crops, KKR_Text, KKR_Soil, KKR_Coord, KKR, Precision_Classes, Recall_Classes, FVal)]
        
        Con = sqlite3.connect(ResDatabase)
        Cur = Con.cursor()
        sql_command = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='TCF'"
        res = Cur.execute(sql_command).fetchall()
        if res[0][0] == 0:
            sql = """
                CREATE TABLE TCF (
                Name String NOT NULL,
                Treshold FLOAT NOT NULL,
                Datatype String NOT NULL,
                Time FLOAT NOT NULL,
                Number_of_Tokens INTEGER NOT NULL,
                Num_of_full_correct_labels INTEGER NOT NULL,
                Num_of_full_correct_relevant_labels INTEGER NOT NULL,
                Num_of_wrong_relevant_labels INTEGER NOT NULL,
                Num_of_all_classes INTEGER NOT NULL,
                Num_of_correct_classes INTEGER NOT NULL,
                Num_of_false_classes INTEGER NOT NULL,
                Calc_Class_Crop INTEGER NOT NULL,
                Calc_Class_Text INTEGER NOT NULL,
                Calc_Class_Soil INTEGER NOT NULL,
                Calc_Class_Coord INTEGER NOT NULL,
                Real_Class_Crop INTEGER NOT NULL,
                Real_Class_Text INTEGER NOT NULL,
                Real_Class_Soil INTEGER NOT NULL,
                Real_Class_Coord INTEGER NOT NULL,
                KKR_Class_Crops FLOAT NOT NULL,
                KKR_Class_Text FLOAT NOT NULL,
                KKR_Class_Soil FLOAT NOT NULL,
                KKR_Class_Coord FLOAT NOT NULL,
                KKR_Class_All FLOAT NOT NULL,
                Precision_Class FLOAT NOT NULL,
                Recall_Class FLOAT NOT NULL,
                FVal FLOAT NOT NULL,
                PRIMARY KEY(Name, Treshold, Datatype)
                );"""
            Cur.execute(sql_command)
            Con.commit()
        sql = "INSERT INTO TCF VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        Cur.executemany(sql, results_list)
        Con.commit()
        Con.close()
