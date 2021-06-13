import os
import random

modeltye = "c" # Only real data

randomseed = "Randomseed"
TrainTestRatio = 70
NumOfEpochs = 3
Batch_Size_Train = 8

Maxlength = 917

import sqlite3

CurDir = os.getcwd()
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()

xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)

random.seed(randomseed)
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

Dataset = []
import Module_Coordinates as mc
Dict = {}
TokenCoords = []
for (FPID, File, Par) in OriginalPars:
    if len(Par)<Maxlength:
        (Six, Eight, NF, E) = mc.find_coordinates(Par)
        
        for (Converted_Coords, Found_Coord, Par) in Six + Eight:
            splittedpar = split_string(Par)
            FullLabels = []
            TokenedCoords = Tokenizer.tokenize(split_string(Found_Coord))
            if len(Converted_Coords) == 6:
                Grad1 = split_string(Converted_Coords[0])
                Min1 = split_string(Converted_Coords[1])
                Lat = Converted_Coords[2]
                Grad2 = split_string(Converted_Coords[3])
                Min2 = split_string(Converted_Coords[4])
                Long = Converted_Coords[5]
                Labels = []
                for i in range(len(TokenedCoords)):
                    Labels.append(0)
                for i in range(len(TokenedCoords)-1):
                    if TokenedCoords[i] == Grad1[0]:
                        if len(Grad1)>1:
                            if TokenedCoords[i+1] == Grad1[1]:
                                Labels[i+1] = 1
                                Labels[i] = 1
                            else:
                                pass
                        else:
                            Labels[i] = 1
                        
                    if TokenedCoords[i] == Min1[0]:
                        if len(Min1)>1:
                            if TokenedCoords[i+1] == Min1[1]:
                                Labels[i] = 2
                                Labels[i+1] = 2
                        else:
                            Labels[i] = 2
                    if TokenedCoords[i] == Lat:
                        Labels[i] = 4
                    if TokenedCoords[i] == Grad2[0]:
                        if len(Grad2)>1:
                            if TokenedCoords[i+1] == Grad2[1]:
                                Labels[i] = 1
                                Labels[i+1] = 1
                        else:
                            Labels[i] = 1
                    if TokenedCoords[i] == Min2[0]:
                        if len(Min2)>1:
                            if TokenedCoords[i+1] == Min2[1]:
                                Labels[i] = 2
                                Labels[i+1] = 2
                        else:
                            Labels[i] = 2
                    if TokenedCoords[i] == Long:
                        Labels[i] = 5
            else:
                Grad1 = split_string(Converted_Coords[0])
                Min1 = split_string(Converted_Coords[1])
                Sek1 = split_string(Converted_Coords[2])
                Lat = Converted_Coords[3]
                Grad2 = split_string(Converted_Coords[4])
                Min2 = split_string(Converted_Coords[5])
                Sek2 = split_string(Converted_Coords[6])
                Long = Converted_Coords[7]
                Labels = []
                
                for i in range(len(TokenedCoords)):
                    Labels.append(6)
                for i in range(len(TokenedCoords)-1):
                    if TokenedCoords[i] == Grad1[0]:
                        if len(Grad1)>1:
                            if TokenedCoords[i+1] == Grad1[1]:
                                Labels[i] = 1
                                Labels[i+1] = 1
                        else:
                            Labels[i] = 1
                    if TokenedCoords[i] == Min1[0]:
                        if len(Min1) > 1:
                            if TokenedCoords[i+1] == Min1[1]:
                                Labels[i] = 2
                                Labels[i+1] = 2
                        else:
                            Labels[i] = 2
                    if TokenedCoords[i] == Sek1[0]:
                        if len(Sek1) > 1:
                            if TokenedCoords[i+1] == Sek1[1]:
                                Labels[i] = 3
                                Labels[i+1] = 3
                        else:
                            Labels[i] = 3
                    if TokenedCoords[i] == Lat:
                        Labels[i] = 4
                    if TokenedCoords[i] == Grad2[0] :
                        if len(Grad2)>1:
                            if TokenedCoords[i+1] == Grad2[1]:
                                Labels[i] = 1
                                Labels[i+1] = 1
                    if TokenedCoords[i] == Min2[0]:
                        if len(Min2)>1:
                            if TokenedCoords[i+1] == Min2[1]:
                                Labels[i] = 2
                                Labels[i+1] = 2
                        else:
                            Labels[i] = 2
                    if TokenedCoords[i] == Sek2[0]:
                        if len(Sek2)>1:
                            if TokenedCoords[i+1] == Sek2[1]:
                                Labels[i] = 3
                                Labels[i+1] = 3
                        else:
                            Labels[i] = 3
                    if TokenedCoords[i] == Long:
                        Labels[i] = 5                
            TokenPar = Tokenizer.tokenize(splittedpar)
            Found = False
            
            for i in range(len(TokenPar)):
                if TokenPar[i] == TokenedCoords[0] and not Found:
                    Found = True
                    for j in range(0, len(Labels)):
                        if TokenPar[i+j] != TokenedCoords[j]:
                            Found = False
                if Found:
                    Starting_i = i
                    break
            

            
            for i in range(0, Starting_i):
                FullLabels.append(0)

            FullLabels = FullLabels + Labels

            for i in range(Starting_i + len(Labels), len(TokenPar)):
                FullLabels.append(0)
            if len(FullLabels) != len(TokenPar):
                print("Error")
            else:
                Dataset.append((FullLabels, splittedpar))

                    


NumOfTraining = int(len(Dataset)/100*TrainTestRatio)




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


random.shuffle(Dataset)
Training_Data_ne = []
Training_Labels = []
Test_Data_ne = []
Test_Labels = []
All_Labels = [-100]
for (Labels, Splitted_Par) in Dataset:
    for Label in Labels:
        if Label not in All_Labels:
            All_Labels.append(Label)
    if len(Training_Data_ne)<NumOfTraining:
        Training_Data_ne.append(Splitted_Par)    
        Training_Labels.append([-100] + Labels + [-100]) # CLS / SEP
    else:
        Test_Data_ne.append(Splitted_Par)
        Test_Labels.append([-100] + Labels + [-100]) # CLS / SEP

NumOfLabels = len(All_Labels)
 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Training_Data = Tokenizer(Training_Data_ne, padding = True)
Test_Data = Tokenizer(Test_Data_ne, padding = True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        labelcopy = self.labels[idx]
        for i in range(len(item['input_ids'])-len(self.labels[idx])):
            labelcopy.append(-100)
        item['labels'] = torch.tensor(labelcopy)
        return item

    def __len__(self):
        return len(self.labels)

Train_Dataset = Dataset(Training_Data, Training_Labels)
Test_Dataset = Dataset(Test_Data, Test_Labels)

from transformers import DistilBertForTokenClassification
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=8)


from transformers import Trainer, TrainingArguments



Batch_Size_Eval = Batch_Size_Train

training_args = TrainingArguments(
    output_dir= CurDir + '/Results/Outputs/',          # output directory
    num_train_epochs=NumOfEpochs,              # total number of training epochs
    per_device_train_batch_size=Batch_Size_Train,  # batch size per device during training
    per_device_eval_batch_size=Batch_Size_Eval,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir= CurDir + '/Results/Logs/',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=Train_Dataset,         # training dataset
    eval_dataset=Test_Dataset             # evaluation dataset
)

trainer.train()

ModName = "TC1c_Model_Coordinates"
model.save_pretrained(CurDir + "/Models/" + ModName)
print("Saved model")
