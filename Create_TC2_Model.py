

NumOfEpochs = 3
Maxlength = 917
Training_Split = 70
randomseed = "Randomseeeeeeeed"



import random
random.seed(randomseed)
import os
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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dataset = []
import Module_Coordinates as mc
Dict = {}
TokenCoords = []
for (FPID, File, Par) in OriginalPars:
    if len(Par)<Maxlength:
        (Six, Eight, NF, E) = mc.find_coordinates(Par)
        Found_Coords = []
        for (Converted_Coords, Found_Coord, Par) in Six + Eight:
            Found_Coords.append(Found_Coord)
        TokenPar = Tokenizer.tokenize(Par)
        if len(Found_Coords)>0:
            TokenCoords = []
            for Coords in Found_Coords:
                TokenCoords.append(Tokenizer.tokenize(Coords))
        ParLabels = []
        for i in range(len(TokenPar)):
            ParLabels.append(0)
        for TokenCoord in TokenCoords:
            Inserted = False
            for i in range(len(TokenPar)):
                Found = True
                if not Inserted:
                    if TokenPar[i] == TokenCoord[0]:
                        for j in range(len(TokenCoord)):
                            if Found and i+j<len(TokenPar):
                                if TokenPar[i+j] != TokenCoord[j]:
                                    Found = False
                            else:
                                Found = False
                        if Found:
                            for k in range(len(TokenCoord)):
                                ParLabels[i+k] = 1
                            Inserted = True
        ParLabels = [-100] + ParLabels + [-100] # CLS + SEP
        Dataset.append((ParLabels, Par))
        
random.shuffle(Dataset)


Training_Labels = []
Test_Labels = []
Training_Data_n = []
Test_Data_n = []

for (Labels, Par) in Dataset:
    if len(Training_Labels) <= (len(Dataset)/100*70):
        Training_Data_n.append(Par)
        Training_Labels.append(Labels)
    else:
        Test_Data_n.append(Par)
        Test_Labels.append(Labels)

Training_Data = Tokenizer(Training_Data_n, padding = True)
Test_Data = Tokenizer(Test_Data_n, padding = True)


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
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


from transformers import Trainer, TrainingArguments


Batch_Size_Train = 8
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
ModName = "TC2_Model_Coordinates/"
model.save_pretrained(CurDir + "/Models/" + ModName)
print("Saved model")
