import os
import time
CurDir = os.getcwd()
Filepath =  CurDir + "/Files/"

ModName = "Crops"
from transformers import BertTokenizerFast
Tokenizer = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
num_labels = 4 # Crop, Soil, Texture, Coordinate
import torch
from transformers import BertForTokenClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import sqlite3
if not os.path.exists(CurDir + "/Models/" + ModName + "/" + "Parameters.pickle"):


    # ID, Filename, Par
    import Module_Coordinates as mc
    
    
    Basemodel = "bert-base-cased"
    No_Class = [float(0), float(0), float(0), float(0)]
    Crops_Class = No_Class.copy()
    Crops_Class[0] = float(1)


    TestPercentage = 90
    Randomseed = "Final_TC_Alpha"

    import random
    random.seed(Randomseed)

    with open(CurDir + "/Files/TCF_A_Training.pickle", "rb") as file:
        TrainingDataOriginal = pickle.load(file)
    with open(CurDir + "/Files/TCF_A_Test.pickle", "rb") as file:
        TestDataOriginal = pickle.load(file)
    with open(CurDir + "/Files/TCF_A_All.pickle", "rb") as file:
        FittingDataOriginal = pickle.load(file)
    TrainingData = []
    TestData = []
    FittingData = []
    OriginalData = [(TrainingDataOriginal, "Train"), (TestDataOriginal, "Test"), (FittingDataOriginal, "All")]
    for (oData, oDType) in OriginalData:
        for Par_Orig, Labels_Orig, (ParCrops, ParTextures, ParSoils, ParCords) in oData:
            Labels = []
            for Label in Labels_Orig:
                if Label == No_Class or Label == Crops_Class:
                    Labels.append(Label)
                else:
                    Labels.append(No_Class)
            if oDType == "Train":
                TrainingData.append((Par_Orig, Labels, (ParCrops, ParTextures, ParSoils, ParCords)))
            else:
                if oDType == "Test":
                    TestData.append((Par_Orig, Labels, (ParCrops, ParTextures, ParSoils, ParCords)))
                else:
                    if oDType == "All":
                        FittingData.append((Par_Orig, Labels, (ParCrops, ParTextures, ParSoils, ParCords)))
                    else:
                        print("Error in Label correction")
                        print(oDType)
                        input()
    with open(CurDir + "/Files/TCF_Crops_Training.pickle", "wb") as file:
        pickle.dump(TrainingData, file)
    with open(CurDir + "/Files/TCF_Crops_Test.pickle", "wb") as file:
        pickle.dump(TestData, file)
    with open(CurDir + "/Files/TCF_Crops_All.pickle", "wb") as file:
        pickle.dump(FittingData, file)                        
    from transformers import AdamW
    
    from torch.utils.data import DataLoader
    
    Learning_Rate = 5e-5
    
    Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=num_labels).to(device)
    optim = AdamW(Model.parameters(), lr=Learning_Rate)
    PadLength = 510
    DatasetLength = 10000 # 10000
    class Dataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            random.shuffle(TrainingData)
            Par_DS, Labels_DS, (ParCrops, ParTextures, ParSoils, ParCords) = TrainingData[0]
            Labels_DS = Labels_DS.copy()
            TokenizedPar_WithCLSandSEP = Tokenizer(Par_DS)
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

    

    Batch_Size = 8
    Pos_Weight_Vector = torch.ones([num_labels])
    Training_Loader = DataLoader(Dataset(), batch_size = Batch_Size)
    BCEWLL = torch.nn.BCEWithLogitsLoss(pos_weight = Pos_Weight_Vector).to(device)
    Loss_History = []
    Counter = 0
    Time_For_Batch = []
    Custom_Loss = 0
    Stoptime = 28800 # 28800
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

    Model.eval()
    Model.save_pretrained(CurDir + "/Models/" + ModName)
    HistoryOutputPlace = CurDir + "/Results/TCF_Loss/"
    if not os.path.isdir(HistoryOutputPlace):
        os.mkdir(HistoryOutputPlace)
    with open(HistoryOutputPlace + ModName + ".pickle", "wb") as file:
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
else:
    Model = BertForTokenClassification.from_pretrained(CurDir + "/Models/" + ModName, num_labels=num_labels).to(device)
    with open(CurDir + "/Files/TCF_Crops_Training.pickle", "rb") as file:
        TrainingData = pickle.load(file)
    with open(CurDir + "/Files/TCF_Crops_Test.pickle", "rb") as file:
        TestData = pickle.load(file)
    with open(CurDir + "/Files/TCF_Crops_All.pickle", "rb") as file:
        FittingData = pickle.load(file)
Tresholds = [float(0.4), float(0.5), float(0.6), float(0.7), float(0.8), float(0.9), float(0.95), float(0.99), float(0.999), float(0.9999), float(0.99999), float(0.999999)]
Daten = [(TrainingData, "Train"), (TestData, "Test"), (FittingData, "All")]


ResDatabase = CurDir + "/Results/Results.db"

for (Dater, DescriptorData) in Daten:
    for Treshold in Tresholds:
        print("Starting " + DescriptorData + " with treshold " + str(Treshold))
        Starttime = time.time()
        Classes_C1_R1 = {}
        Classes_C0_R1 = {}
        Classes_C1_R0 = {}
        Classes_C0_R0 = {}
        Num_Of_All_Tokens = 0
        Num_Of_All_Relevant_Tokens = 0
        Correct_Labels = 0
        Correct_Relevant_Labels = 0
        Num_Of_Labels_With_n_Ones = {}
        Real_Ones = {}
        for i in range(num_labels):
            Classes_C1_R1[i] = 0
            Classes_C0_R1[i] = 0
            Classes_C1_R0[i] = 0
            Classes_C0_R0[i] = 0
            Real_Ones[i] = 0
            Num_Of_Labels_With_n_Ones[i] = 0
        Num_Of_Labels_With_n_Ones[4] = 0
        for CountAna, (Par_Aus, Labels_Aus, (ParCrops, ParTextures, ParSoils, ParCords)) in enumerate(Dater):
            if CountAna%1000==0:
                print(DescriptorData + " - " + str(Treshold) + " | " + str(CountAna) + "/" + str(len(Dater)))
            Labels_Aus = Labels_Aus.copy()
            Par_Aus = str(Par_Aus)
            Calc_Labels_Par = []
            TokenizedPar = Tokenizer.tokenize(Par_Aus)
            StrEnc = Tokenizer(Par_Aus, return_tensors="pt").to(device)
            Output = Model(**StrEnc)
            Logits = Output.logits[0][1:-1].sigmoid()
            for i in range(len(Labels_Aus)):
                Current_Token = TokenizedPar[i]
                Current_Logits = Logits[i]
                Calc_Label = []
                for value in Current_Logits:
                    if value.item() > Treshold:
                        Calc_Label.append(float(1))
                    else:
                        Calc_Label.append(float(0))
                Calc_Labels_Par.append(Calc_Label)
            # Labels_Aus
            # Calc_Labels_Par
            # TokenizedPar
            Num_Of_All_Tokens += len(Labels_Aus)
            for i in range(len(Labels_Aus)):
                Num_Of_Labels_With_n_Ones[Calc_Labels_Par[i].count(float(1))] += 1
                for j in range(num_labels):
                    if Labels_Aus[i][j] == float(1):
                        Real_Ones[j] += 1
                if float(1) in Labels_Aus[i]:
                    Num_Of_All_Relevant_Tokens += 1
                    if Labels_Aus[i] == Calc_Labels_Par[i]:
                        Correct_Relevant_Labels += 1
                for j in range(num_labels): # Crops, Texture, Soil, Coordinate
                    if Labels_Aus[i][j] == float(0):
                        if Calc_Labels_Par[i][j] == float(0):
                            Classes_C0_R0[j] += 1 # True Negative
                        else:
                            Classes_C1_R0[j] += 1 # False Positive
                    else:
                        if Calc_Labels_Par[i][j] == float(0):
                            Classes_C0_R1[j] += 1 # False Negative
                        else:
                            Classes_C1_R1[j] += 1 # True Positive
                
                if Labels_Aus[i] == Calc_Labels_Par[i]:
                    Correct_Labels += 1
            
            
        Endtime = time.time()
        FullTime = Endtime - Starttime
        Precision = {}
        Recall = {}
        FVal = {}
        Mean_FVal = 0
        for i in range(num_labels):
            if Classes_C1_R1[i]+Classes_C1_R0[i] != 0:
                Precision[i] = Classes_C1_R1[i]/(Classes_C1_R1[i]+Classes_C1_R0[i])
            else:
                Precision[i] = 0
            if Classes_C1_R1[i]+Classes_C0_R1[i] != 0:
                Recall[i] = Classes_C1_R1[i]/(Classes_C1_R1[i]+Classes_C0_R1[i])
            else:
                Recall[i] = 0
            if Precision[i] + Recall[i] != 0:
                FVal[i] = (2*Precision[i]*Recall[i])/(Precision[i]+Recall[i])
            else:
                FVal[i] = 0
            Mean_FVal += FVal[i]
        Mean_FVal = Mean_FVal/num_labels
        
        results_list = [(ModName, Treshold, DescriptorData, FullTime,
                         Num_Of_All_Tokens, Correct_Labels, Num_Of_All_Relevant_Tokens, Correct_Relevant_Labels, #8
                         Real_Ones[0], Classes_C1_R1[0], Classes_C0_R0[0], Classes_C0_R1[0], Classes_C1_R0[0], 
                         Real_Ones[1], Classes_C1_R1[1], Classes_C0_R0[1], Classes_C0_R1[1], Classes_C1_R0[1],
                         Real_Ones[2], Classes_C1_R1[2], Classes_C0_R0[2], Classes_C0_R1[2], Classes_C1_R0[2], 
                         Real_Ones[3], Classes_C1_R1[3], Classes_C0_R0[3], Classes_C0_R1[3], Classes_C1_R0[3], # 20
                         Precision[0], Recall[0], FVal[0],
                         Precision[1], Recall[1], FVal[1],
                         Precision[2], Recall[2], FVal[2],
                         Precision[3], Recall[3], FVal[3], # 12
                         Mean_FVal # 1
                         )]
        
        Con = sqlite3.connect(ResDatabase)
        Cur = Con.cursor()
        sql_command = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='TCF'"
        res = Cur.execute(sql_command).fetchall()
        if res[0][0] == 0:
            print("Creating table")
            sql = """
                CREATE TABLE TCF (
                Name String NOT NULL,
                Treshold FLOAT NOT NULL,
                Datatype String NOT NULL,
                Time FLOAT NOT NULL,
                Num_Of_All_Tokens INTEGER NOT NULL,
                Num_Of_Correct_Labels INTEGER NOT NULL,
                Num_Of_All_Relevant_Tokens INTEGER NOT NULL,
                Num_Of_All_Correct_Relevant_Labels INTEGER NOT NULL,
                Crops_Real INTEGER NOT NULL,
                Crops_TP INTEGER NOT NULL,
                Crops_TN INTEGER NOT NULL,
                Crops_FN INTEGER NOT NULL,
                Crops_FP INTEGER NOT NULL,
                Textures_Real INTEGER NOT NULL,
                Texture_TP INTEGER NOT NULL,
                Texture_TN INTEGER NOT NULL,
                Texture_FN INTEGER NOT NULL,
                Texture_FP INTEGER NOT NULL,
                Soils_Real INTEGER NOT NULL,
                Soils_TP INTEGER NOT NULL,
                Soils_TN INTEGER NOT NULL,
                Soils_FN INTEGER NOT NULL,
                Soils_FP INTEGER NOT NULL,
                Coords_Real INTEGER NOT NULL,
                Coords_TP INTEGER NOT NULL,
                Coords_TN INTEGER NOT NULL,
                Coords_FN INTEGER NOT NULL,
                Coords_FP INTEGER NOT NULL,
                Crop_Prec FLOAT NOT NULL,
                Crop_Recall FLOAT NOT NULL,
                Crop_F FLOAT NOT NULL,
                Texture_Prec FLOAT NOT NULL,
                Texture_Recall FLOAT NOT NULL,
                Texture_F FLOAT NOT NULL,
                Soils_Prec FLOAT NOT NULL,
                Soils_Recall FLOAT NOT NULL,
                Soils_F FLOAT NOT NULL,
                Coords_Prec FLOAT NOT NULL,
                Coords_Recall FLOAT NOT NULL,
                Coords_F FLOAT NOT NULL,
                Mean_F FLOAT NOT NULL,
                PRIMARY KEY(Name, Treshold, Datatype)
                );"""
            Cur.execute(sql)
            Con.commit()
        sql = "INSERT INTO TCF VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        Cur.executemany(sql, results_list)
        Con.commit()
        Con.close()
print("Finished")
