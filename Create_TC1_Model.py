import pickle
import os
import random


randomseed = "Randomseed"
TrainTestRatio = 70
NumOfEpochs = 3
Batch_Size_Train = 8

CurDir = os.getcwd()
FakeDataFile = open(CurDir + "/Files/FakeData.pickle", "rb")
Dataset = pickle.load(FakeDataFile)
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

random.seed(randomseed)
random.shuffle(Dataset)
Training_Data_ne = []
Training_Labels = []
Test_Data_ne = []
Test_Labels = []
All_Labels = []
for ((ID, Par), Labels) in Dataset:
    NewLabels = [-100]
    for Label in Labels:
        NewLabels.append(LabelDict[Label])
        if Label not in All_Labels:
            All_Labels.append(Label)
    NewLabels.append(-100)
    if len(Training_Data_ne)<NumOfTraining:
        Training_Data_ne.append(Par)    
        Training_Labels.append(NewLabels)
    else:
        Test_Data_ne.append(Par)
        Test_Labels.append(NewLabels)

NumOfLabels = len(All_Labels)

if len(LabelDict.keys()) != NumOfLabels+3:
    print("Label Error")


from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)


        
    

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


ModName = "TC1_Model_Coordinates_Fake/"
model.save_pretrained(CurDir + "/Models/" + ModName)
print("Saved model")
