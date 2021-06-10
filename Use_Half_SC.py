# Variables

ModelType = "Soils"
#ModelType = "Soilless"
#ModelType = "Coordinates"





def get_label(Str, Model, Tokenizer):
    StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
    Output = Model(**StrEnc)
    Softmaxed = Output.logits.softmax(1)
    Label = Softmaxed[0][1].item()
    return Label

def stitch_together(ListOfStrings):
    Orig = ""
    for Str in ListOfStrings:
        Orig = Orig + Str + " "
    return Orig[:-1]

def split_check(Str, Model, Tokenizer, Perc):
    StrSplit = Str.split(" ")
    Part1Split = StrSplit[:int(len(StrSplit)/2)]
    Part2Split = StrSplit[int(len(StrSplit)/2):]
    Part1 = stitch_together(Part1Split)
    Part2 = stitch_together(Part2Split)
    L1 = get_label(Part1, Model, Tokenizer)
    L2 = get_label(Part2, Model, Tokenizer)
    Returnwords = []
    Returnlist = []
    if L1 > Perc:
        if len(Part1.split(" ")) == 1:
            Returnwords.append(Part1)
        else:
            Returnlist = split_check(Part1, Model, Tokenizer, Perc)
            for word in Returnlist:
                Returnwords.append(word)
    else:
        if L2 < Perc:
            Returnwords.append(Str)
    if L2 > Perc:
        if len(Part2.split(" ")) == 1:
            Returnwords.append(Part2)
        else:
            Returnlist = split_check(Part2, Model, Tokenizer, Perc)
            for word in Returnlist:
                Returnwords.append(word)
    return Returnwords


if ModelType == "Coordinates":
    import re
    def coord_regex(Stringlist):
        # Find xx°xx'N
        Found = False
        rege1 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,10}[0-9][0-9].{1,4}(?:W|E).{1,1}"
        # Find xx°xx'xx''N
        rege2 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,6}[0-9][0-9].{1,4}(?:W|E).{1,1}"
        regelist = [rege1, rege2]
        for potcoord in Stringlist:
            for regel in regelist:
                results = re.findall(regel, potcoord)
                if len(results)>0:
                    Found = True
        return Found    

import os
import sqlite3

CurDir = os.getcwd()



Paragraphs = {}
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
if ModelType == "Soils" or ModelType == "Soilless":
    PercentageO = 90
    

    xs = "Select FPID, Soil, SoilD FROM Soils"
    SoilPars = Cur.execute(xs).fetchall()
    xs = "Select * FROM Pars"
    Pars = Cur.execute(xs).fetchall()
    Con.close()


    for (ParID, Soil, SoilD) in SoilPars:
        if ParID not in Paragraphs.keys():
            Paragraphs[ParID] = [(Soil, SoilD)]
        else:
            Paragraphs[ParID].append((Soil, SoilD))
            
    

if ModelType == "Coordinates":
    xs = "Select FPID, NumOfCoords from Coordinates"
    CoordsPars = Cur.execute(xs).fetchall()
    xs = "Select * FROM Pars"
    Pars = Cur.execute(xs).fetchall()
    Con.close()
    for (ParID, NumOfCoords) in CoordsPars:
        Paragraphs[ParID] = NumOfCoords
    PercentageO = 70

ModPath = CurDir + "/Models/"
Model_Path = ModPath + "SC_Model_" + ModelType + "/"
    

import torch
import transformers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Model = transformers.DistilBertForSequenceClassification.from_pretrained(Model_Path, output_attentions = True)
Model.eval()
Model.to(device)



from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)

Maxlength = 917



Dataset = []
LongPars = []
for (FPID, File, Par) in Pars:
    if len(Par) < Maxlength:
        if FPID in Paragraphs.keys():
            Dataset.append(((FPID, Par), 1))
        else:
            Dataset.append(((FPID, Par), 0))
            Paragraphs[FPID] = []
    else:
        LongPars.append(Par)

Percentage = PercentageO/100

Resultsdict = {}

o1 = "NullEins"
o0 = "NullNull"
B1F = "B1SoilFound"
B1N = "B1SoilNotF"
B0F = "B0SoilFound"
B0N = "B0SoilNotF"


Resultsdict[11] = 0 # Bewertet 1, Label 1
Resultsdict[10] = 0 # Bewertet 1, Label 0
Resultsdict[o1] = 0
Resultsdict[o0] = 0
Resultsdict[B1F] = 0
Resultsdict[B1N] = 0
Resultsdict[B0F] = 0
Resultsdict[B0N] = 0



       
    
    
Status = 0
for ((FPID, Par), Label) in Dataset:
    CalcLabel = get_label(Par, Model, Tokenizer)
    OverPerc = False
    if CalcLabel > Percentage:
        OverPerc = True
    if Label == 1:
        if CalcLabel > Percentage:
            Resultsdict[11]+=1
        else:
            Resultsdict[o1]+=1
    else:
        if CalcLabel > Percentage:
            Resultsdict[10]+=1
        else:
            Resultsdict[o0]+=1
    if Label == 1:
        Found_Words = split_check(Par, Model, Tokenizer, Percentage)
        if ModelType == "Soils" or ModelType == "Soilless":
            for (Soil, SoilD) in Paragraphs[FPID]:
                Found = False
                soil = Soil.lower()
                soilD = SoilD.lower()
                for Fword in Found_Words:
                    fword = Fword.lower()
                    if fword in soil or soil in fword or soilD in fword or fword in soilD:
                        Found = True
                if Found == True:
                    if OverPerc:
                        Resultsdict[B1F] += 1
                    else:
                        Resultsdict[B0F] += 1
                else:
                    if OverPerc:
                        Resultsdict[B1N] += 1
                    else:
                        Resultsdict[B0N] += 1
        if ModelType == "Coordinates":
            for i in range(Paragraphs[FPID]):
                Found = coord_regex(Found_Words)
                if Found == True:
                    if OverPerc:
                        Resultsdict[B1F] += 1
                    else:
                        Resultsdict[B0F] += 1
                else:
                    if OverPerc:
                        Resultsdict[B1N] += 1
                    else:
                        Resultsdict[B0N] += 1                    
            
    Status+=1
    if Status % 100 == 0:
        print(str(Status) + " von " + str(len(Dataset)))

ModName = "Half_" + ModelType
results_list = []
results_list.append((ModName, Resultsdict[11], Resultsdict[10], Resultsdict[o1], Resultsdict[o0],
                          Resultsdict[B1F], Resultsdict[B1N], Resultsdict[B0F], Resultsdict[B0N], PercentageO)) 


Database = CurDir + "/Results/Results.db"
if not os.path.isfile(Database):
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()
    sql_command = """
            CREATE TABLE Results (
            Model String NOT NULL,
            Bew_1_Tat_1 INTEGER NOT NULL,
            Bew_1_Tat_0 INTEGER NOT NULL,
            Bew_0_Tat_1 INTEGER NOT NULL,
            Bew_0_Tat_0 INTEGER NOT NULL,
            B1_Soil_Found INTEGER NOT NULL,
            B1_Soil_Not_Found INTEGER NOT NULL,
            B0_Soil_Found INTEGER NOT NULL,
            B0_Soil_Not_Found INTEGER NOT NULL,
            Percentage NOT NULL,
            PRIMARY KEY(Model, Percentage)
            );"""
    Cur.execute(sql_command)
    Con.commit()
    Con.close()

Con = sqlite3.connect(Database)
Cur = Con.cursor()
sql_command = "INSERT INTO Results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
Cur.executemany(sql_command, results_list)
Con.commit()
Con.close()
print("Finished")
