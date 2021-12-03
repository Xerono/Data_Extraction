# https://huggingface.co/transformers/custom_datasets.html

# Extract prelabeled Dataset and prepreprocess
# Variables
#   Modeltype
#   PreTrainedModel
#   MaxLength is for the Tokenizer


for ModelType in ["Soils_dc", "Soilless_dc", "Coordinates_dc"]:

    PreTrainedModel = "distilbert-base-cased"
    Maxlength = 917

    # Variables for Training:
    #   NumOfEpochs
    #   Batch Size

    NumOfEpochs = 3
    Batch_Size_Train = 8


    # Variables for Traning/Testset creation:
    #   Ratio: Training/Test-Split on Dataset
    #   Seed: Seed for shuffled list


    Ratio = 0.25
    Seed = "Shuffleseed"


    import sqlite3
    import os

    CurDir = os.getcwd()

    Database = CurDir + "/Files/Database.db"
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()

    Paragraphs = {}

    if ModelType == "Soils_dc" or ModelType == "Soilless_dc":
        xs = "Select FPID, Soil, SoilD FROM Soils"
        SoilPars = Cur.execute(xs).fetchall()
        xs = "Select * FROM Pars"
        Pars = Cur.execute(xs).fetchall()

        for (ParID, Soil, SoilD) in SoilPars:
            if ParID not in Paragraphs.keys():
                Paragraphs[ParID] = [(Soil, SoilD)]
            else:
                Paragraphs[ParID].append((Soil, SoilD))
                
    if ModelType == "Coordinates_dc":
        xs = "Select FPID, NumOfCoords FROM Coordinates"
        LocPars = Cur.execute(xs).fetchall()
        xs = "Select * FROM Pars"
        Pars = Cur.execute(xs).fetchall()
        for (ParID, NumOfCoords) in LocPars:
            Paragraphs[ParID] = NumOfCoords
        
    Con.close()



    Dataset = []
    LongPars = []

    def remove_soils(FPID, Par):
        rpar = Par
        Soils = Paragraphs[FPID]
        for (Soil, SoilD) in Soils:
            soil = Soil.lower()
            soilD = SoilD.lower()
            rpar = rpar.replace(soil, "")
            rpar = rpar.replace(soilD, "")
        return rpar

    for (FPID, File, Par) in Pars:
        Par = Par.lower()
        if len(Par) < Maxlength:
            if FPID in Paragraphs.keys():
                if ModelType == "Soilless_dc":
                    Par = remove_soils(FPID, Par)
                Dataset.append(((FPID, Par), 1))
            else:
                Dataset.append(((FPID, Par), 0))
                Paragraphs[FPID] = []
        else:
            LongPars.append(Par)


    # Split into training/testdata
               

    import random

    Training_Text = []
    Training_Label = []
    Test_Text = []
    Test_Label = []

    random.seed(Seed)
    random.shuffle(Dataset)

    NumOfTraining = int(len(Dataset)-len(Dataset)*Ratio)
    NumOfTest = len(Dataset)-NumOfTraining


    for ((FPID, Par), Label) in Dataset:
        if len(Training_Text)<=NumOfTraining:
            Training_Text.append(Par)
            Training_Label.append(Label)
        else:
            Test_Text.append(Par)
            Test_Label.append(Label)

    # Tokenization
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import DistilBertTokenizerFast
    Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)

    Train_Encodings = Tokenizer(Training_Text, padding = True)
    Test_Encodings = Tokenizer(Test_Text, padding = True)


    # Datasettype

    

    class Datasetloader(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    Train_Dataset = Datasetloader(Train_Encodings, Training_Label)
    Test_Dataset = Datasetloader(Test_Encodings, Test_Label)

    # Training

    from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

    Batch_Size_Eval = Batch_Size_Train

    training_args = TrainingArguments(
        output_dir= CurDir + '/Results/',          # output directory
        num_train_epochs=NumOfEpochs,              # total number of training epochs
        per_device_train_batch_size=Batch_Size_Train,  # batch size per device during training
        per_device_eval_batch_size=Batch_Size_Eval,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir= CurDir + '/Results/Logs/',            # directory for storing logs
        logging_steps=10,
    )
    
    model = DistilBertForSequenceClassification.from_pretrained(PreTrainedModel).to(device)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=Train_Dataset,         # training dataset
        eval_dataset=Test_Dataset             # evaluation dataset
    )

    trainer.train()

    # Save Model
    model.eval()

    ModName = "SC_Model_" + ModelType + "/"

    model.save_pretrained(CurDir + "/Models/" + ModName)
    print("Saved model " + ModelType)
