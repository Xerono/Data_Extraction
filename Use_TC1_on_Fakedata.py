import os

# a/b:
# a: Coordinates get replaced in existing sentences
# b: Coordinates get inserted in sentences with random words

# d/b:
# d: Distilbert
# b: Bert

# u/c:
# u: Uncased
# c: Cased
potmodels = ["a_du", "a_dc", "a_bc", "a_bu", "b_du", "b_dc", "b_bc", "b_bu"]
all_dicts = {}
for modeltype in potmodels:
    CurDir = os.getcwd()

    ModPath = CurDir + "/Models/"
    Model_Path = ModPath + "TC1" + modeltype + "_Model_Coordinates/"

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if modeltype == "a_du" or modeltype == "b_du":
        PreTrainedModel = 'distilbert-base-uncased'
        from transformers import DistilBertTokenizerFast
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
        from transformers import DistilBertForTokenClassification
        model = DistilBertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)

    if modeltype == "a_dc" or modeltype == "b_dc":
        PreTrainedModel = 'distilbert-base-cased'
        from transformers import DistilBertTokenizerFast
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
        from transformers import DistilBertForTokenClassification
        model = DistilBertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
        
    if modeltype == "a_bc" or modeltype == "b_bc":
        PreTrainedModel = 'bert-base-cased'
        from transformers import BertTokenizerFast
        Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
        from transformers import DistilBertForTokenClassification
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)
    if modeltype == "a_bu" or modeltype == "b_bu":
        PreTrainedModel = 'bert-base-uncased'
        from transformers import BertTokenizerFast
        Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
        from transformers import BertForTokenClassification
        model = BertForTokenClassification.from_pretrained(Model_Path, num_labels=8).to(device)


    import sqlite3
    import pickle
    FakeDataFile = open(CurDir + "/Files/FakeData_" + modeltype + ".pickle", "rb")
    Dataset = pickle.load(FakeDataFile)
    OriginalPars = []
    for ((ID, Splitpar), Labels) in Dataset:
        OriginalPars.append((ID, Labels, Splitpar))

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
    Model.eval()


    NotFound = []

    Data = []

    for (FPID, Label, Par) in OriginalPars:
        Splitpar = Tokenizer.tokenize(Par)
        Grads = []
        Mins = []
        Sek = []
        Long = []
        Lat = []
        for i in range(len(Label)):
            if Label[i] == "Grad":
                Grads.append(Splitpar[i])
            if Label[i] == "Min":
                Mins.append(Splitpar[i])
            if Label[i] == "Sek":
                Sek.append(Splitpar[i])
            if Label[i] == "Latitude":
                Lat.append(Splitpar[i])
            if Label[i] == "Longitude":
                Long.append(Splitpar[i])
        if len(Grads) == 4 and len(Mins) == 4 and len(Lat)==1 and len(Long) == 1:
            if len(Sek)==4:
                PotCoord = ((Grads[0] + Grads[1], Mins[0] + Mins[1], Sek[0] + Sek[1], Lat[0], Grads[2] + Grads[3], Mins[2] + Mins[3], Sek[2] + Sek[3], Long[0]))
                Data.append((PotCoord, 8, Par))
            else:
                PotCoord = ((Grads[0] + Grads[1], Mins[0] + Mins[1], Lat[0], Grads[2] + Grads[3], Mins[2] + Mins[3], Long[0]))
                Data.append((PotCoord, 6, Par))
        else:
            NotFound.append(Par)

    Dataset = []
    Numbers = [0,0,0]
    for (Coords, Num, SplitPar) in Data:
        if len(SplitPar) < Maxlength:
            Dataset.append((Coords, Num, SplitPar))
            if Num == 6:
                Numbers[0] += 1
            else:
                Numbers[1] += 1
    for SplitPar in NotFound:
        if len(SplitPar) < Maxlength:
            Dataset.append(([], 0, SplitPar))
            Numbers[2] += 1
    print(Numbers)
            
    def get_label(Str, Model, Tokenizer):
        StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
        Output = Model(**StrEnc)
        Softmaxed = Output.logits.softmax(1)
        Labels = []
        for labels in Softmaxed[0]:
            lblcur = []
            for lbl in labels:
                lblcur.append(lbl.item())
            Labels.append(lblcur)
        return Labels[1:-1] # CLS and SEP


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

    for (PotCords, LenCoords, SplitPar) in Dataset:
        if len(SplitPar)<Maxlength:
            Tokens = Tokenizer.tokenize(SplitPar)
            Labels = get_label(SplitPar, Model, Tokenizer)
            Classes = get_token_class(Tokens, Labels)
            RevTokens = extract_relevant_classes(Tokens, Classes)

            if LenCoords == 0:
                if len(RevTokens)>0:
                    Resultsdict[10] += 1
                else:
                    Resultsdict[o0] += 1
            else:
                if len(RevTokens)>0:
                    Resultsdict[11] += 1
                else:
                    Resultsdict[o1] += 1

                (GradE, MinE, SekE, DirE, rnum) = ToCoords(RevTokens)
                if rnum != LenCoords:
                    if len(RevTokens)>=0:
                        Resultsdict[B1N] += 1
                    else:
                        Resultsdict[B0N] += 1
                else:
                    ReturnCoords = []
                    Found = True
                    for i in range(LenCoords):
                        ReturnCoords.append(False)
                    Hits = 0
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
                            if (not ReturnCoords[3]) and dire == PotCords[3]:
                                ReturnCoords[3] = dire
                            if (not ReturnCoords[7]) and dire == PotCords[7]:
                                ReturnCoords[7] = dire
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
                            if (not ReturnCoords[2]) and dire == PotCords[2]:
                                ReturnCoords[2] = dire
                            if (not ReturnCoords[5]) and dire == PotCords[5]:
                                ReturnCoords[5] = dire
                    Hits = 0

                    for i in range(LenCoords):
                        if ReturnCoords[i] == PotCords[i]:
                            Hits += 1
                        else:
                            Found = False
                    HitDict[Hits] += 1
                    if Found:
                        Resultsdict[B1F] += 1
                    else:
                        Resultsdict[B1N] += 1
            Runner+=1
            if Runner%1000==0:
                print(str(Runner) + "/" + str(len(Dataset)))


            
    results_list = []

    ModName = "TC1" + modeltype + "_Fakedata"

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
    sql_command = "INSERT INTO Results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    Cur.executemany(sql_command, results_list)
    Con.commit()
    Con.close()
    print(modeltype)
    print("Fakedatadict:")
    print(HitDict)
    all_dicts[modeltype] = HitDict
    print("Finished")
print(all_dicts)        

