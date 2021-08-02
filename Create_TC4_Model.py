Storage = False

def create(Inputs):
    print(Inputs)
    (Cutting_Pars, CoordsToNoise, Delete_Coords, Detailed_Labels) = Inputs
    Cut_Par = bool(int(Cutting_Pars))
    Coord_To_Noise = bool(int(CoordsToNoise))
    Delete_Teilcoords = bool(int(Delete_Coords))
    Detailed_Labels = bool(int(Detailed_Labels))
    Code = ""
    for var in Inputs:
        Code += str(var)
    
    if Detailed_Labels:
        num_labels = 12
    else:
        num_labels = 9
    
    Basemodel = "bert-base-cased"
    #Basemodel = "distilbert-base-uncased"

    PadLength = 320
    DatasetLength = 10000 # Datasetlength / Batch size = Iterations per Epoch
    Stoptime = 28800 # 8 hours
    Batch_Size_Train = 8
    Learning_Rate = 5e-5
    Custom_Loss = 0.1
    TestPercentage = 10

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
    
    import random
    random.seed(Randomseed)

    import Module_Coordinates as mc

    PwC = []

    for (FPID, File, Par) in OriginalPars:
        (Six, Eight, NF, E) = mc.find_coordinates(Par)
        Found_Coords = Six + Eight
        Coords = []
        if len(Found_Coords)>0 and len(mc.split_string(Par))<MaxLength:
            for (PotCord, StringCord, Par) in Found_Coords:
                Coords.append((PotCord, StringCord))
            PwC.append((Par, Coords))

    Alle_Daten = len(PwC)
    TestDataLength = int(Alle_Daten/100*TestPercentage)
    Testdd = []
    Trainingdd = []

    for entry in PwC:
        if len(Testdd)< TestDataLength:
            Testdd.append(entry)
        else:
            Trainingdd.append(entry)

    import pickle
    with open(CurDir + "/Files/TC4_Training.pickle", "wb") as file:
        pickle.dump(Trainingdd, file)
    with open(CurDir + "/Files/TC4_Test.pickle", "wb") as file:
        pickle.dump(Testdd, file)
    PwC = Trainingdd



    import torch


    from transformers import BertTokenizerFast
    from transformers import BertForTokenClassification
    from transformers import AdamW
    from torch.utils.data import DataLoader

    Symbols = ["•", "H", "V", "¢", ".", "j", "J", "°", ",", ";", "Њ", "Ј", "U",
                   '"', "″", "'", "o", "@", "؇", "-", "¶", "(", ")", "Љ", "±",
                   ":", "µ", "/",
                   "8", "9"] # Found by trial & error
    
    int_to_label = {}
    int_to_label[0] = "Padded"
    int_to_label[1] = "Irrelevant"
    int_to_label[2] = "Noise"
    int_to_label[3] = "Coord"
    int_to_label[4] = "Grad1"
    int_to_label[5] = "Min1"
    int_to_label[6] = "Sek1"
    int_to_label[7] = "Lat"
    int_to_label[8] = "Long"
    int_to_label[9] = "Grad2"
    int_to_label[10] = "Min2"
    int_to_label[11] = "Sek2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=num_labels).to(device)
    Tokenizer = BertTokenizerFast.from_pretrained(Basemodel)
    optim = AdamW(Model.parameters(), lr=Learning_Rate)


    
    def Labelvector_To_Label(Labels):
        Max = -1
        for i in range(len(Labels)):
            if Labels[i]>=Max:
                Max = Labels[i]
        Maxes = []
        for i in range(len(Labels)):
            if Labels[i] == Max:
                Maxes.append(i)
                
        Labels_For_This_Token = []
        for i in Maxes:
            Labels_For_This_Token.append(int_to_label[i])
        return Labels_For_This_Token

    def generate_noise():
        Basic_Label = []     
        for i in range(num_labels):
            Basic_Label.append(float(0))
                            # Padded, Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long

                            # Padded, Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
        Noise = ""
        Labels = []
        Basic_Label[2] = float(1)
        for i in range(random.choice([1,2])):
            Noise = Noise + random.choice(Symbols)
            CurLabel = Basic_Label.copy()
            Labels.append(CurLabel)
            
        return (Noise, Labels)

    def generate_coords():
       
        PotCoords = []
        Labels = []
        CoordsString = ""
        Basic_Label = []
        EightCoords = random.choice([True, False])
        
     # Padded, Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
        for i in range(num_labels):
            Basic_Label.append(float(0))
     # Padded, Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long
            

        Grad1 = str(random.randint(0, 90))
        for i in range(len(Grad1)):
            Cur = Basic_Label.copy()
            Cur[3] = float(1)
            Cur[4] = float(1)
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Grad1 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)

        Min1 = str(random.randint(0, 60))
        for i in range(len(Min1)):
            Cur = Basic_Label.copy()
            Cur[3] = float(1)
            Cur[5] = float(1)
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Min1 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)
            
        if EightCoords:
            Sek1 = str(random.randint(0, 60))
            for i in range(len(Sek1)):
                Cur = Basic_Label.copy()
                Cur[3] = float(1)
                Cur[6] = float(1)
                Labels.append(Cur)
            (Noise, NLabels) = generate_noise()
            CoordsString = CoordsString + Sek1 + Noise
            for NoiseLabels in NLabels:
                Labels.append(NoiseLabels)

        Lat = random.choice(["N", "S"])
        Cur = Basic_Label.copy()
        Cur[3] = float(1)
        Cur[7] = float(1)
        Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Lat + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)

        Grad2 = str(random.randint(0, 90))
        for i in range(len(Grad2)):
            Cur = Basic_Label.copy()
            Cur[3] = float(1)
            if Detailed_Labels:
                Cur[9] = float(1)
            else:
                Cur[4] = float(1)
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Grad2 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)
            
        Min2 = str(random.randint(0, 60))
        for i in range(len(Min2)):
            Cur = Basic_Label.copy()
            Cur[3] = float(1)
            if Detailed_Labels:
                Cur[10] = float(1)
            else:
                Cur[5] = float(1)
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Min2 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)
            
        if EightCoords:
            Sek2 = str(random.randint(0, 60))
            for i in range(len(Sek2)):
                Cur = Basic_Label.copy()
                Cur[3] = float(1)
                if Detailed_Labels:
                    Cur[11] = float(1)
                else:
                    Cur[6] = float(1)
                Labels.append(Cur)
            (Noise, NLabels) = generate_noise()
            CoordsString = CoordsString + Sek2 + Noise
            for NoiseLabels in NLabels:
                Labels.append(NoiseLabels)

        Lon = random.choice(["W", "E"])
        Cur = Basic_Label.copy()
        Cur[3] = float(1)
        Cur[8] = float(1)
        Labels.append(Cur)
        CoordsString = CoordsString + Lon

        if EightCoords:
            PotCoords = (Grad1, Min1, Sek1, Lat, Grad2, Min2, Sek2, Lon)
        else:
            PotCoords = (Grad1, Min1, Lat, Grad2, Min2, Lon)
        return(PotCoords, CoordsString, Labels)

            
    
        
    def Replace(ParCord):
        Basic_Label = []
        for i in range(num_labels):
            Basic_Label.append(float(0))

        Basic_Label[1] = float(1)
        (Par, ListOfCoords) = ParCord
        FullNewCoords = []
        for (Coord, String) in ListOfCoords:
            (PotCoords, CoordsString, Labels) = generate_coords()
            Par = Par.replace(String, CoordsString)
            FullNewCoords.append((PotCoords, CoordsString, Labels))
        Coords_In_This_Par = []
        FullLabels = []
        SplitPar = mc.split_string(Par)
        TokenizedSplitPar = Tokenizer.tokenize(SplitPar)
        for i in range(len(TokenizedSplitPar)):
            FullLabels.append(Basic_Label.copy())
        for (PotCoords, CoordsString, Labels) in FullNewCoords:
            CoordsSplit = Tokenizer.tokenize(mc.split_string(CoordsString))
            for i in range(0, len(TokenizedSplitPar) - len(CoordsSplit) + 1):
                if TokenizedSplitPar[i:i+len(CoordsSplit)] == CoordsSplit:
                    for j in range(len(Labels)):
                        FullLabels[i+j] = Labels[j]
            Coords_In_This_Par.append((PotCoords, CoordsString))
        return(SplitPar, FullLabels, Coords_In_This_Par)

    class Dataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            global Storage
            Irrel_Label = []
            Padded_Label = []
             # Padded, Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
            for i in range(num_labels):
                Irrel_Label.append(float(0))
                Padded_Label.append(float(0))
 # Padded, Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long

            Irrel_Label[1] = float(1)
            Padded_Label[0] = float(1)
            random.shuffle(PwC)
            (Current_Par, CordList) = PwC[0]
            ECoords = []
            SCoords = []
            for (ECord, SCord) in CordList:
                ECoords.append(ECord)
                SCoords.append(SCord)
            if Cut_Par:
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

            (SP, Labels, CoordsInPar) = Replace((Current_Par, CordList))

            
            if Coord_To_Noise:
                if not Storage:
                    if random.choice([1,2,3,4]) == 1:
                        NoiseLabels = []
                        NoisePar = SP
                        for (Coords, StringC) in CoordsInPar:
                            newc = list(StringC)
                            random.shuffle(newc)
                            newcs = ""
                            for teilstring in newc:
                                newcs += teilstring
                            NoisePar = NoisePar.replace(mc.split_string(StringC), mc.split_string(newcs))
                        NoisePar = mc.split_string(NoisePar)
                        TokenizedNP = Tokenizer.tokenize(NoisePar)
                        for i in range(len(TokenizedNP)):
                            NoiseLabels.append(Irrel_Label.copy())
                        Storage = (NoisePar, NoiseLabels)
                else:
                    (SP, Labels) = Storage
                    Storage = False
            

            TSP = Tokenizer(SP)
            
            TSPcoded = TSP['input_ids'][1:-1]# CLS / SEP

            if Delete_Teilcoords:
                if random.choice([1,2,3,4]) == 1:
                    Slices_With_Coords = []
                    for i in range(1, len(Labels)-1):
                        if Labels[i][3] == float(1):
                            Slices_With_Coords.append(i)
                    if Slices_With_Coords:
                        To_Delete = [] # Deleting on lists while traversing the list is bothersome
                        for i in Slices_With_Coords:
                            if random.choice([1,2,3,4]) == 1:
                                To_Delete.append(i)
                        for i in sorted(To_Delete, reverse=True):
                            del TSPcoded[i]
                        if To_Delete:
                            Labels = []
                            for i in range(len(TSPcoded)):
                                Labels.append(Irrel_Label.copy())
            
            for i in range(PadLength-len(Labels)):
                TSPcoded.append(0)
                Labels.append(Padded_Label.copy())
 
            item = {}
            item['input_ids'] =  torch.tensor(TSPcoded)
            item['labels'] = torch.tensor(Labels)
            return item

        def __len__(self):
            return DatasetLength
    
    import time
    TrainData = Dataset()
    Training_Loader = DataLoader(TrainData, batch_size = Batch_Size_Train)
    BCEWLL = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones([num_labels])).to(device)
    Loss_History = []
    Counter = 0
    Starttime = time.time()
    while time.time() - Starttime < Stoptime:
        for batch in Training_Loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            Output = Model(input_ids)
            Logits = Output.logits
            Loss = BCEWLL(Logits, labels)
            lossnum = Loss.item()
            Loss_History.append(lossnum)
            Loss.backward()
            optim.step()
            if Counter % 1000 == 0:
                print(Code + " with loss of " + str(round(lossnum, 6)) + " (" + str(Counter) + " steps, " + str(round(time.time() - Starttime, 2)) + "/" + str(Stoptime) + " seconds)")
            Counter += 1
    endtime = time.time()
    FullTime = endtime - Starttime
    mdl = "TC4_" + Code
    ModName = mdl + "_Model/"
    Model.save_pretrained(CurDir + "/Models/" + ModName)

    import pickle
    HistoryOutputPlace = CurDir + "/Results/TC4_Loss/" 
    if not os.path.isdir(HistoryOutputPlace):
        os.mkdir(HistoryOutputPlace)
    with open(HistoryOutputPlace + Code + ".pickle", "wb") as file:
        pickle.dump(Loss_History, file)

    Parameters["FullTime"] = FullTime
    Parameters["Len_los_history"] = len(Loss_History)
        
    with open(CurDir + "/Models/" + ModName + "/" + "Parameters.pickle", "wb") as file:
        pickle.dump(Parameters, file)
    print("Model " + mdl + " saved.")
