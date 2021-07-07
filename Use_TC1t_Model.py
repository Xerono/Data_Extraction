import os
import pickle

Debug = False
modelsorig = ["dc", "bc"]

for modeltype in modelsorig:
    CurDir = os.getcwd()

    ModPath = CurDir + "/Models/"

    Model_Path = ModPath + "TC1t_" + modeltype + "_Model_Coordinates/"

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modeltype == "bc":
        PreTrainedModel = 'bert-base-cased'
        from transformers import BertTokenizerFast
        Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
        from transformers import BertForTokenClassification
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
    if modeltype == "dc":
        PreTrainedModel = 'distilbert-base-cased'
        from transformers import DistilBertTokenizerFast
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
        from transformers import DistilBertForTokenClassification
        model = DistilBertForTokenClassification.from_pretrained(PreTrainedModel, num_labels=8).to(device)

    import sqlite3
    Database = CurDir + "/Files/Database.db"
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()
    xs = "Select * FROM Pars"
    OriginalPars = Cur.execute(xs).fetchall()
    Con.close()

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


    Maxlength = 917
    model.eval()

    import re
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

    Sixers = []
    Eighters = []
    Errors = []
    NotFound = []
    import Module_Coordinates as mc
    for (FPID, File, Par) in OriginalPars:
        (Six, Eight, NE, E) = mc.find_coordinates(Par)
        for el in Six:
            Sixers.append(el)
        for el in Eight:
            Eighters.append(el)
        for el in NE:
            NotFound.append(el)
        for el in E:
            Errors.append(el)




    Dataset = []
    NumbersTrue = [0,0,0]
    for (Coords, Regex, SplitPar) in Sixers:
        if len(SplitPar) < Maxlength:
            Dataset.append((Coords, 6, split_string(SplitPar)))
            NumbersTrue[0] += 1
    for (Coords, Regex, SplitPar) in Eighters:
        if len(SplitPar) < Maxlength:
            Dataset.append((Coords, 8, split_string(SplitPar)))
            NumbersTrue[1] += 1
    for SplitPar in NotFound:
        if len(SplitPar) < Maxlength:
            Dataset.append(([], 0, split_string(SplitPar)))
            NumbersTrue[2] += 1


    Datasets = []
    Datasets.append((Dataset, "Real"))

    
    FakeDataFile = open(CurDir + "/Files/FakeData_t_" + modeltype + ".pickle", "rb")
    
    Dataset = []
    NumbersFake = [0,0,0]
    DatasetPre = pickle.load(FakeDataFile)
    NumNum = DatasetPre[0]
    DatasetPre = DatasetPre[1:]
    print(NumNum)
    for ((ID, Splitpar), PotCoords, Labels) in DatasetPre:
        Dataset.append((PotCoords, len(PotCoords), Splitpar))
        if len(PotCoords) == 6:
            NumbersFake[0] += 1
        else:
            if len(PotCoords) == 8:
                NumbersFake[1] += 1
            else:
                NumbersFake[2] += 1
    Datasets.append((Dataset, "Fake"))
    print(NumbersFake)
            
    def get_label(Str, Model, Tokenizer):
        StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
        Output = Model(**StrEnc)
        Softmaxed = Output.logits.softmax(-1)
        Labels = []
        for labels in Softmaxed[0]:
            lblcur = []
            for lbl in labels:
                lblcur.append(lbl.item())
            Labels.append(lblcur)
        return Labels[1:-1] # CLS + SEP


    def get_token_class(Tokens, Labels):
        Classes = []
        for i in range(len(Tokens)):
            MaxClass = max(Labels[i])
            CurClasses = []
            for j in range(len(Labels[i])):
                if Labels[i][j] == MaxClass:
                    CurClasses.append(j)
            Classes.append(CurClasses)
        return Classes

    def extract_relevant_classes(Tokens, Classes):
        Relevant = []
        for i in range(len(Tokens)):
            for Element in Classes[i]:
                PotClasses = []
                if Element in [1, 2, 3, 4, 5]:
                    PotClasses.append(Element)
            Relevant.append((Tokens[i], PotClasses))
        return Relevant


    def Extend(List):
        RList = []
        for item in List:
            for item2 in List:
                RList.append(str(item) + str(item2))
        RList = List + RList
        return RList

    def ToCoords(RevTokens):
        rnum = 0
        Grad = []
        Min = []
        Sek = []
        Lat = []
        Long = []
        for (Token, ClassList) in RevTokens:
            if "#" not in Token:
                for Class in ClassList:
                    if Class == 1:
                        Grad.append(Token)
                    if Class == 2:
                        Min.append(Token)
                    if Class == 3:
                        Sek.append(Token)
                    if Class == 4:
                        Lat.append(Token)
                    if Class == 5:
                        Long.append(Token)
        GradE = Extend(Grad)
        MinE = Extend(Min)
        SekE = Extend(Sek)
        LatE = Extend(Lat)
        LongE = Extend(Long)
        DirE = Extend(LatE+LongE)
        if len(SekE)>0:
            rnum = 8
        else:
            rnum = 6
        return(GradE, MinE, SekE, DirE, rnum)


    for (Dataset, RealOrFake) in Datasets:
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

        Runner = 0
        HitDict = {}
        for i in range(9):
            HitDict[i] = 0
        print("Starting with " + RealOrFake)
        for (PotCords, LenCoords, SplitPar) in Dataset:
            if len(SplitPar)<Maxlength:
                Tokens = Tokenizer.tokenize(SplitPar)
                Labels = get_label(SplitPar, model, Tokenizer)
                Classes = get_token_class(Tokens, Labels)
                RevTokens = extract_relevant_classes(Tokens, Classes)

                FoundRele = False
                for (Token, Labellist) in RevTokens:
                    for lbl in Labellist:
                        FoundRele = True

                if LenCoords == 0:
                    if FoundRele:
                        Resultsdict[10] += 1
                    else:
                        Resultsdict[o0] += 1
                else:
                    if FoundRele:
                        Resultsdict[11] += 1
                    else:
                        Resultsdict[o1] += 1

                    (GradE, MinE, SekE, DirE, rnum) = ToCoords(RevTokens)
                    ReturnCoords = []
                    for i in range(LenCoords):
                        ReturnCoords.append(False)
                    if LenCoords == 8:
                        for grad in GradE:
                            if (not ReturnCoords[0]) and grad == PotCords[0]:
                                ReturnCoords[0] = grad
                            if (not ReturnCoords[4]) and grad == PotCords[4]:
                                ReturnCoords[4] = grad
                        for mint in MinE:
                            if (not ReturnCoords[1]) and mint == PotCords[1]:
                                ReturnCoords[1] = mint
                            if (not ReturnCoords[5]) and mint == PotCords[5]:
                                ReturnCoords[5] = mint
                        for sek in SekE:
                            if (not ReturnCoords[2]) and sek == PotCords[2]:
                                ReturnCoords[2] = sek
                            if (not ReturnCoords[6]) and sek == PotCords[6]:
                                ReturnCoords[6] = sek
                        for dire in DirE:
                            diru = dire.upper()
                            if (not ReturnCoords[3]) and diru == PotCords[3]:
                                ReturnCoords[3] = diru
                            if (not ReturnCoords[7]) and diru == PotCords[7]:
                                ReturnCoords[7] = diru
                    else:
                        for grad in GradE:
                            if (not ReturnCoords[0]) and grad == PotCords[0]:
                                ReturnCoords[0] = grad
                            if (not ReturnCoords[3]) and grad == PotCords[3]:
                                ReturnCoords[3] = grad
                        for mint in MinE:
                            if (not ReturnCoords[1]) and mint == PotCords[1]:
                                ReturnCoords[1] = mint
                            if (not ReturnCoords[4]) and mint == PotCords[4]:
                                ReturnCoords[4] = mint
                        for dire in DirE:
                            diru = dire.upper()
                            if (not ReturnCoords[2]) and diru == PotCords[2]:
                                ReturnCoords[2] = diru
                            if (not ReturnCoords[5]) and diru == PotCords[5]:
                                ReturnCoords[5] = diru
                    Hits = 0
                    Found = True
                    for i in range(len(ReturnCoords)):
                        if ReturnCoords[i] == PotCords[i]:
                            Hits += 1
                        else:
                            Found = False
                    HitDict[Hits] += 1
                    if Found:
                        Resultsdict[B1F] += 1               
                    else:
                        if FoundRele:
                            Resultsdict[B1N] += 1
                        else:
                            Resultsdict[B0N] += 1
                Runner+=1
                if Debug and FoundRele:
                    for (Token, Labellist) in RevTokens:
                        for lbl in Labellist:
                            print(Token + " | " + str(IntLabelDict[lbl]))
                    print(PotCords)
                    print(ReturnCoords)
                    print(SplitPar)
                    input()
                if Runner%20000==0:
                    print(str(Runner) + "/" + str(len(Dataset)))
                    pass
                    


                
        results_list = []


        ModName = "TC1t_" + modeltype + "_" + RealOrFake + "data"


        results_list.append((ModName, Resultsdict[11], Resultsdict[10], Resultsdict[o1], Resultsdict[o0]
                             , Resultsdict[B1F], Resultsdict[B1N], Resultsdict[B0F], Resultsdict[B0N], 100)) 

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

        sql_command = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='HitDicts'"
        res = Cur.execute(sql_command).fetchall()
        if res[0][0] == 0:
            sql_command = """
                    CREATE TABLE HitDicts (
                    Model String NOT NULL,
                    Zero INTEGER NOT NULL,
                    Eins INTEGER NOT NULL,
                    Zwei INTEGER NOT NULL,
                    Drei INTEGER NOT NULL,
                    Vier INTEGER NOT NULL,
                    Fuenf INTEGER NOT NULL,
                    Sechs INTEGER NOT NULL,
                    Sieben INTEGER NOT NULL,
                    Acht INTEGER NOT NULL,
                    Sixers INTEGER NOT NULL,
                    Eighters INTEGER NOT NULL,
                    Empty INTEGER NOT NULL,
                    PRIMARY KEY(Model)
                    );"""
            Cur.execute(sql_command)
            Con.commit()
            
        sql_command = "INSERT INTO Results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        Cur.executemany(sql_command, results_list)
        Con.commit()
        sql_command = "INSERT INTO HitDicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        Numbers = [0, 0, 0]
        if RealOrFake == "Real":
            Numbers = NumbersTrue
        else:
            Numbers = NumbersFake
        results_list = [("t_" + modeltype + "_" + RealOrFake, HitDict[0], HitDict[1], HitDict[2], HitDict[3], HitDict[4], HitDict[5], HitDict[6], HitDict[7], HitDict[8], Numbers[0], Numbers[1], Numbers[2])]
        Cur.executemany(sql_command, results_list)
        Con.commit()
        Con.close()

        print("Finished " + RealOrFake)

            

