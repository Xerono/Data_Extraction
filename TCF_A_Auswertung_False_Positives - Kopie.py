from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
import torch
import os
import pickle



CurDir = os.getcwd()
No_Class = [float(0), float(0), float(0), float(0)]
Crops_Class = No_Class.copy()
Crops_Class[0] = float(1)
Texture_Class = No_Class.copy()
Texture_Class[1] = float(1)
Soils_Class = No_Class.copy()
Soils_Class[2] = float(1)
Coords_Class = No_Class.copy()
Coords_Class[3] = float(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 4
Model = BertForTokenClassification.from_pretrained(CurDir + "/Models/Coords_CustomTrain", num_labels=num_labels).to(device)
Tokenizer = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
with open(CurDir + "/Files/TCF_Coords_CustomTrain_All.pickle", "rb") as file:
    FittingData = pickle.load(file)
with open(CurDir + "/Files/TCF_Coords_CustomTrain_Training.pickle", "rb") as file:
    TrainingData = pickle.load(file)
with open(CurDir + "/Files/TCF_Coords_CustomTrain_Test.pickle", "rb") as file:
    TestData = pickle.load(file)

Treshold = 0.8

import random
random.shuffle(FittingData)


print("Datatype? 1 2 3 4")
dtype = int(input())
if dtype == 1:
    Int_Class = Crops_Class
if dtype == 2:
    Int_Class = Texture_Class
if dtype == 3:
    Int_Class = Soils_Class
if dtype == 4:
    Int_Class = Coords_Class
for Entry in FittingData:
    Par_Aus, Labels_Aus, LabeledData = Entry

    Calc_Labels_Par = []
    TokenizedPar = Tokenizer.tokenize(Par_Aus)
    StrEnc = Tokenizer(Par_Aus, return_tensors="pt").to(device)
    Output = Model(**StrEnc)
    Logits = Output.logits[0][1:-1].sigmoid()
    SomethingFound = []
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
    FaPoFound = []
    for i in range(len(Calc_Labels_Par)):
        if Calc_Labels_Par[i] == Int_Class and Labels_Aus[i] != Int_Class:
            FaPoFound.append(i)
    if FaPoFound:
        print(Par_Aus)
        print(LabeledData[dtype-1])
        print()
        for i in range(len(Labels_Aus)):
            if i in FaPoFound:
                print("FaPo: " + str(Calc_Labels_Par[i]) + "   -   " + str(TokenizedPar[i]))
            else:
                print(str(Calc_Labels_Par[i]) + "   -   " + str(TokenizedPar[i]))
        input()

