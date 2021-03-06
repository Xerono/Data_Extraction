Storage = False

def create(Inputs):
    (CPar1, CTN1, DT1, CLO1, DL1) = Inputs

    Cut_Par = bool(int(CPar1))
    Coord_To_Noise = bool(int(CTN1))
    Delete_Teilcoords = bool(int(DT1))
    Custom_LossO = bool(int(CLO1))
    Detailed_Labels = bool(int(DL1))

    print(Inputs)


    Basemodel = "distilbert-base-uncased"
    Basemodel = "bert-base-cased"


    PadLength = 450
    DatasetLength = 10000 # Datasetlength / Batch size = Iterations per Epoch
    Stoptime = 28800 # 8 hours
    Batch_Size_Train = 8
    Learning_Rate = 5e-5
    Custom_Loss = 0.1
    TestPercentage = 10


    import random
    import time
    Randomseed = "DasIstEinSeed"

    Parameters = {}
    Parameters["Basemodel"] = Basemodel
    Parameters["Randomseed"] = Randomseed
    Parameters["PadLength"] = PadLength
    Parameters["DatasetLength"] = DatasetLength
    Parameters["Stoptime"] = Stoptime
    Parameters["Batch_Size_Train"] = Batch_Size_Train
    Parameters["Learning_Rate"] = Learning_Rate
    Parameters["Custom_Loss"] = Custom_Loss
    Parameters["TestPercentage"] = TestPercentage




    import os
    import sqlite3

    CurDir = os.getcwd()
    Database = CurDir + "/Files/Database.db"
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()
    MaxLength = 917
    xs = "Select * FROM Pars"
    OriginalPars = Cur.execute(xs).fetchall()
    Con.close()

    random.seed(Randomseed)




    import Module_Coordinates as mc

    All_Paragraphs = []

    for (FPID, File, Par) in OriginalPars:
        (Six, Eight, NF, E) = mc.find_coordinates(Par)
        Found_Coords = Six + Eight
        Coords = []
        if len(Found_Coords)>0:
            for (PotCord, StringCord, Par) in Found_Coords:
                Coords.append((PotCord, StringCord))
            if len(mc.split_string(Par))<MaxLength:
                All_Paragraphs.append((Par, Coords))

    LabelDict, IntToLabel = mc.labels_to_int()

    import torch


    from transformers import BertTokenizerFast
    from transformers import BertForTokenClassification



    def get_cord_labels(Coord, SString):
        TokenString = Tokenizer.tokenize(SString)
        Labels = []
        PossibleLabels = []
        for i in range(len(Coord)):
            PossibleLabels.append(0)
        
        for i in range(len(TokenString)):
            Labels.append(-1)
        if Detailed_Labels:# Grad1, Min1, Sek1, Lat, Grad2, Min2, Sek2, Long, Noise, Irrelevant, Pad
            if len(Coord) == 6:
                PossibleLabels[0] = 0
                PossibleLabels[1] = 1
                PossibleLabels[2] = 3
                PossibleLabels[3] = 4
                PossibleLabels[4] = 5
                PossibleLabels[5] = 7
            else:
                PossibleLabels[0] = 0
                PossibleLabels[1] = 1
                PossibleLabels[2] = 2
                PossibleLabels[3] = 3
                PossibleLabels[4] = 4
                PossibleLabels[5] = 5
                PossibleLabels[6] = 6
                PossibleLabels[7] = 7
        else:# Grad, Min, Sek, Lat, Long, Noise, Irrelevant, Pad
            if len(Coord) == 6:
                PossibleLabels[0] = 0
                PossibleLabels[1] = 1
                PossibleLabels[2] = 3
                PossibleLabels[3] = 0
                PossibleLabels[4] = 1
                PossibleLabels[5] = 4
            else:
                PossibleLabels[0] = 0
                PossibleLabels[1] = 1
                PossibleLabels[2] = 2
                PossibleLabels[3] = 3
                PossibleLabels[4] = 0
                PossibleLabels[5] = 1
                PossibleLabels[6] = 2
                PossibleLabels[7] = 4
        for i in range(len(Coord)):
            crd = Coord[i]
            TokenCord = Tokenizer.tokenize(mc.split_string(crd))
            TokenCord_Following = TokenCord.copy()
            TokenCord_Following[0] = "##" + TokenCord_Following[0]
            for j in range(len(TokenString)-len(TokenCord)+1):
                if TokenString[j:j+len(TokenCord)] == TokenCord or TokenString[j:j+len(TokenCord)] == TokenCord_Following:
                    for k in range(len(TokenCord)):
                        if Labels[j+k] == -1:
                            Labels[j+k] = PossibleLabels[i]
        for i in range(len(Labels)):
            if Labels[i] == -1: # "Not Labels[i]" => 0 = False
                if Detailed_Labels:
                    Labels[i] = 8 # Noise
                else:
                    Labels[i] = 5 # Noise
        return (SString, Coord, Labels)

    def get_par_labels(ParCord):
        (Par, ListOfCoords, Detailed_Labels) = ParCord
        FullNewCoords = []
        NewCoords = []
        for (Coord, String) in ListOfCoords:
            (CoordsString, PotCords, Labels) = get_cord_labels(Coord, mc.split_string(String))
            NewCoords.append(CoordsString)
            FullNewCoords.append((CoordsString, PotCords, Labels))
        FullLabels = []
        SplitPar = mc.split_string(Par)
        TokenizedSplitPar = Tokenizer.tokenize(SplitPar)
        for i in range(len(TokenizedSplitPar)):
            if Detailed_Labels:
                FullLabels.append(9) # Irrelevant
            else:
                FullLabels.append(6) # Irrelevant
        for (CoordsString, PotCords, Labellist) in FullNewCoords:
            CoordsSplit = Tokenizer.tokenize(mc.split_string(CoordsString))
            for i in range(0, len(TokenizedSplitPar) - len(CoordsSplit) + 1):
                if TokenizedSplitPar[i:i+len(CoordsSplit)] == CoordsSplit:
                    for j in range(len(Labellist)):
                        FullLabels[i+j] = Labellist[j]
        return (SplitPar, FullLabels, NewCoords)






    class Dataset(torch.utils.data.Dataset):
        
        
        def __getitem__(self, idx):
            global Storage
            random.shuffle(All_Paragraphs)
            (Current_Par, CordList) = All_Paragraphs[0]
            ECoords = []
            SCoords = []
            for (ECord, SCord) in CordList:
                ECoords.append(ECord)
                SCoords.append(SCord)
            if Cut_Par:
                if CordList:
                    #Paragraph before first coordinates
                    FirstCoords = SCoords[0]
                    PrePar = Current_Par.split(FirstCoords)[0]
                    FirstSplit = PrePar.split(" ")
                    Cutnum = random.randint(0, len(FirstSplit))
                    NewPre = FirstSplit[Cutnum:]
                    Pre = ""
                    for Word in NewPre:
                        Pre = Pre + Word + " "
                    #Paragraph after last coordinates
                    LastCoords = SCoords[-1]
                    PostPar = Current_Par.split(LastCoords)[-1]
                    LastSplit = PostPar.split(" ")
                    Cutnum = random.randint(0, len(LastSplit))
                    NewPost = LastSplit[:Cutnum]
                    Post = ""
                    for Word in NewPost:
                        Post = Post + Word + " "
                    Current_Par = Current_Par.replace(PrePar, Pre)
                    Current_Par = Current_Par.replace(PostPar, Post)
                else:
                    splitcurpar = Current_Par.split(" ")
                    firstsplit = splitcurpar[:int(len(splitcurpar)/2)]
                    lastsplit = splitcurpar[int(len(splitcurpar)/2):]
                    splitpre = random.randint(0, len(firstsplit))
                    splitpost = random.randint(0, len(lastsplit))
                    newparlist = firstsplit[splitpre:] + lastsplit[:splitpost]
                    newpar = ""
                    for word in newparlist:
                        newpar = newpar + word + " "
                    Current_Par = newpar[:-1]


            (SP, Labels, NewCoords) = get_par_labels((Current_Par, CordList, Detailed_Labels))

            if Coord_To_Noise:
                if not Storage:
                    if CordList:
                        if random.choice([1,2,3,4]) == 1:
                            NoiseList = []
                            NoisePar = Current_Par
                            for stringcoord in NewCoords:
                                coord_list = list(stringcoord)
                                random.shuffle(coord_list)
                                NewCords = ""
                                for item in coord_list:
                                    NewCords = NewCords + item
                                NoisePar = NoisePar.replace(stringcoord, NewCords)
                            NoisePar = Current_Par
                               
                            NoisePar = mc.split_string(NoisePar)
                            TNP = Tokenizer.tokenize(NoisePar)
                            FullLabels = []
                            for i in range(len(TNP)):
                                FullLabels.append(LabelDict["Nul"])
                            Storage = (NoisePar, FullLabels, NewCoords)

                else:
                    (SP, Labels, NewCoords) = Storage
                    Storage = False
            

            TSP = Tokenizer(SP)
            Attentionmask = []
            for i in range(len(Labels)):
                Attentionmask.append(1)
            
            TSPcoded = TSP['input_ids'][1:-1]# CLS / SEP

            if Delete_Teilcoords:
                if CordList:
                    Slices_With_Coords = []
                    for i in range(1, len(Labels)-1):
                        if Labels[i] in Interesting_Labels:
                            Slices_With_Coords.append(i)
                    if Slices_With_Coords and random.choice([1,2,3,4]) == 1:
                        NumberToDelete = random.randint(1, int(len(Slices_With_Coords)))
                        Deleted = []
                        for i in range(NumberToDelete):
                            ToDelete = random.choice(Slices_With_Coords)
                            while ToDelete in Deleted:
                                ToDelete = random.choice(Slices_With_Coords)
                            Deleted.append(ToDelete)
                        for i in Deleted:
                            if i in range(len(TSPcoded)):
                                del TSPcoded[i]
                                del Labels[i]
                                del Attentionmask[i]
                        for i in range(len(Labels)):
                            if Labels[i] != LabelDict["Nul"]:
                                Labels[i] = LabelDict["Nul"]
            
            for i in range(PadLength-len(Labels)):
                TSPcoded.append(0)
                Labels.append(-100)
                Attentionmask.append(0)

            item = {}
            item['input_ids'] =  torch.tensor(TSPcoded)
            item['labels'] = torch.tensor(Labels)
            item['attention_mask'] = torch.tensor(Attentionmask)
            return item

        def __len__(self):
            return DatasetLength

    TrainData = Dataset()

    from transformers import AdamW
    from torch.utils.data import DataLoader





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import time

    random.seed(Randomseed)
    if Detailed_Labels:
        Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=11) # Grad1, Min1, Sek1, Lat, Grad2, Min2, Sek2, Long, Noise, Irrelevant, Pad
    else:
        Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=8) # Grad, Min, Sek, Lat, Long, Noise, Irrelevant, Pad
    Tokenizer = BertTokenizerFast.from_pretrained(Basemodel)
    optim = AdamW(Model.parameters(), lr=Learning_Rate)
    Training_Loader = DataLoader(TrainData, batch_size = Batch_Size_Train)
    Model.to(device)
    loss_history = []
    convergence = []
    ConvergenceFound = False
    Storage = False
    starttime = time.time()
    Counter = 0
    CLoss_History = []
    Code = str(int(Cut_Par)) + str(int(Coord_To_Noise)) + str(int(Delete_Teilcoords)) + str(int(Custom_LossO)) + str(int(Detailed_Labels))
    if Detailed_Labels:
        Interesting_Labels = [1, 2, 3, 4, 5, 7, 8] # Grad1, Min1, Sek1, Lat, Long, Grad2, Min2
        Missing_Labels = [1, 2, 4, 5, 7, 8]
    else:
        Interesting_Labels = [1, 2, 3, 4, 5] # Grad, Min, Sek, Lat, Long
        Missing_Labels = [1,2,4,5]
    while time.time() - starttime < Stoptime :
        for batch in Training_Loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = Model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            CLossForBatch = 0
            
            if Custom_LossO:
                Current_Par_In_Batch = 0
                Softmaxed = outputs.logits.softmax(-1)
                for SingleParLabels in Softmaxed:
                    ParagraphHasCoordinates  = False
                    for label in labels[Current_Par_In_Batch]:
                        if label.item() in Interesting_Labels:
                            ParagraphHasCoordinates = True
                    
                    LabelsForPar = []
                    for ScoresForOneToken in SingleParLabels:
                        TknMax = 0
                        for i in range(len(ScoresForOneToken)):
                            if ScoresForOneToken[i].item() > TknMax:
                                TknMax = ScoresForOneToken[i].item()
                        TknLabels = []
                        for i in range(len(ScoresForOneToken)):
                            if ScoresForOneToken[i].item() == TknMax:
                                TknLabels.append(i)
                        LabelsForPar.append(TknLabels)
                    WhichCoords = []
                    All_Calculated_Labels = []
                    for lblistpertoken in LabelsForPar:
                        lbl = random.choice(lblistpertoken)
                        All_Calculated_Labels.append(lbl)
                        if lbl in Interesting_Labels:
                            WhichCoords.append(lbl)
                            
                    Add_Loss = 0
                    # If there's an interesting label, it has to be correct
                    for i in range(len(All_Calculated_Labels)):
                        Actual_Label = labels[Current_Par_In_Batch][i].item()
                        Calculated_Label = All_Calculated_Labels[i]
                        if Actual_Label in Interesting_Labels:
                            if Actual_Label != Calculated_Label:
                                Add_Loss += Custom_Loss * 2 # Important, so doubled
                    if ParagraphHasCoordinates:
                        # Something is missing (3s are optional)
                        for Target in Missing_Labels:
                            if Target not in WhichCoords:
                                Add_Loss += Custom_Loss
                        # First Long before Lat
                        if 4 in WhichCoords and 5 in WhichCoords:
                            if WhichCoords.index(4) > WhichCoords.index(5):
                                Add_Loss += Custom_Loss
                    else:
                        # Coordinates found in paragraph without coordinates
                        for item in WhichCoords:
                            Add_Loss += Custom_Loss
                    loss += Add_Loss
                    CLossForBatch += Add_Loss
                    Current_Par_In_Batch += 1
                CLoss_History.append(CLossForBatch)
            loss.backward()
            optim.step()
            lossnum = loss.item()
            loss_history.append(lossnum)       
            if Counter % 1000 == 0:
                print(Code + " with loss of " + str(round(lossnum, 6)) + "(" + str(Counter) + " steps, " + str(round(time.time() - starttime, 2)) + "/" + str(Stoptime) + " seconds)")
            Counter += 1
               
    endtime = time.time()
    FullTime = endtime - starttime

    mdl = "TC5_" + Code
    ModName =  mdl + "_Model/"
    Model.save_pretrained(CurDir + "/Models/" + ModName)

    import pickle
    HistoryOutputPlace = CurDir + "/Results/TC5_Loss/"
    if not os.path.isdir(HistoryOutputPlace):
        os.mkdir(HistoryOutputPlace)
    with open(HistoryOutputPlace + Code + ".pickle", "wb") as file:
        pickle.dump(loss_history, file)
    if Custom_LossO:
        CHOP = CurDir + "/Results/TC5_Custom_Loss/"
        if not os.path.isdir(CHOP):
            os.mkdir(CHOP)
        with open(CHOP + Code + ".pickle", "wb") as file:
            pickle.dump(CLoss_History, file)

    Parameters["FullTime"] = FullTime
    Parameters["Len_los_history"] = len(loss_history)
        
    with open(CurDir + "/Models/" + ModName + "/" + "Parameters.pickle", "wb") as file:
        pickle.dump(Parameters, file)
    print("Model " + mdl + " saved.")


