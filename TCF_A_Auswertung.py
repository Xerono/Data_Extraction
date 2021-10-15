from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
import torch
import os
import pickle



CurDir = os.getcwd()
No_Class = [float(0), float(0), float(0), float(0)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 4
Model = BertForTokenClassification.from_pretrained(CurDir + "/Models/Alpha", num_labels=num_labels).to(device)
Tokenizer = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
with open(CurDir + "/Files/TCF_A_All.pickle", "rb") as file:
    FittingData = pickle.load(file)
with open(CurDir + "/Files/TCF_A_Training.pickle", "rb") as file:
    TrainingData = pickle.load(file)
with open(CurDir + "/Files/TCF_A_Test.pickle", "rb") as file:
    TestData = pickle.load(file)

print("Enter treshold:")
Treshold = float(input())

print("Show uninteresting paragraphs?")
UnintPars = bool(input())


print("Show only testdata?")
OnlyTest = bool(input())




if OnlyTest:
    DataToLookAt = TestData
else:
    DataToLookAt = FittingData
import random
random.shuffle(DataToLookAt)

for Entry in DataToLookAt:
    Par_Aus, Labels_Aus, (ParCrops, ParTextures, ParSoils, ParCords) = Entry

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
    for lbls in Calc_Labels_Par:
        if lbls != No_Class:
            SomethingFound = True
        else:
            SomethingFound = False
    if (UnintPars or ParCrops or ParTextures or ParSoils or ParCords or SomethingFound):
        print(Par_Aus)
        print()
        print("Crops: " + str(ParCrops))
        print("Textures: " + str(ParTextures))
        print("Soils: " + str(ParSoils))
        print("Coords: " + str(ParCords))
        print()
        if Entry in TrainingData:
            print("Trainingdata")
        if Entry in TestData:
            print("Testdata")
        if Entry not in TrainingData and Entry not in TestData:
            print("Neither training nor test")
        print("Found labels:")
        for i in range(len(Calc_Labels_Par)):
            if Calc_Labels_Par[i] != No_Class:
                print(str(Calc_Labels_Par[i]) + " | " + str(TokenizedPar[i]))
        print("_______________________________________")
        print()
        input()
