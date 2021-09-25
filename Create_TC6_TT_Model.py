# Different kind of coordinate generation
Storage = False

def create(Inputs, Cust_Tok):
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
        num_labels = 11
    else:
        num_labels = 8
    if Cust_Tok:
        Clarifier = "CT"
    else:
        Clarifier = "NT"    
    Basemodel = "bert-base-cased"

    DatasetLength = 10000 # Datasetlength / Batch size = Iterations per Epoch
    Stoptime = 28800 # 8 hours
    Batch_Size_Train = 8
    Learning_Rate = 5e-5
    Custom_Loss = 0.1
    TestPercentage = 10
    PadLength = 510
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
    import pickle
    CurDir = os.getcwd()
    Database = CurDir + "/Files/TC6_TT_Training.pickle"
    with open(Database, "rb") as dbf:
        OriginalPars = pickle.load(dbf)

    
    import random
    random.seed(Randomseed)

    import Module_Coordinates as mc

    PwC = []
    All_Coordinates = []
    for (FPID, File, Par) in OriginalPars:
        (Six, Eight, NF, E) = mc.find_coordinates(Par)
        Found_Coords = Six + Eight
        Coords = []
        if len(Found_Coords)>0:
            for (PotCord, StringCord, Par) in Found_Coords:
                Coords.append((PotCord, StringCord))
                All_Coordinates.append((PotCord, StringCord))
            PwC.append((Par, Coords))

    import torch


    from transformers import BertTokenizerFast
    from transformers import BertForTokenClassification
    from transformers import AdamW
    from torch.utils.data import DataLoader

    Symbols = ["•",  "¢", ".", "°", ",", ";",
                   '"', "″", "'",  "@",  "-", "¶", "(", ")",  "±",
                   ":", "µ", "/",
                   "8", "9"] # Found by trial & error
    # Unks have to be removed:
    Unknowns = ["Љ", "؇", "Љ"]
    Problems_in_Creation = [ "H", "V","Ј", "U","j", "J", "o"]
    int_to_label = {}
    int_to_label[0] = "Irrelevant"
    int_to_label[1] = "Noise"
    int_to_label[2] = "Coord"
    int_to_label[3] = "Grad1"
    int_to_label[4] = "Min1"
    int_to_label[5] = "Sek1"
    int_to_label[6] = "Lat"
    int_to_label[7] = "Long"
    int_to_label[8] = "Grad2"
    int_to_label[9] = "Min2"
    int_to_label[10] = "Sek2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=num_labels).to(device)

    if Cust_Tok:
        Tokenizer = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
    else:
        Tokenizer = BertTokenizerFast.from_pretrained(Basemodel)
    optim = AdamW(Model.parameters(), lr=Learning_Rate)
    PwCCopy = []
    for (Par, CL) in PwC:
        if len(Tokenizer.tokenize(Par))<PadLength:
               PwCCopy.append((Par, CL))
    PwC = PwCCopy

    # New in CT
    for sym in Symbols:
        if Tokenizer.tokenize(sym) == ['[UNK]']:
            print("Error:")
            print(sym)
            print("tokenizes to '[UNK]'")
            input()
    # End New in CT


    Pos_Weight_Vector = torch.ones([num_labels])
    
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


    # Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long
    # Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2

    def get_Labels_for_coords(TCord, CordL, SCord):
        Zero_Label = []
        for i in range(num_labels):
            Zero_Label.append(float(0))
        Label_Grad1 = Zero_Label.copy()
        Label_Grad1[2] = float(1)
        Label_Grad1[3] = float(1)

        Label_Min1 = Zero_Label.copy()
        Label_Min1[2] = float(1)
        Label_Min1[4] = float(1)

        Label_Sek1 = Zero_Label.copy()
        Label_Sek1[2] = float(1)
        Label_Sek1[5] = float(1)

        Label_Lat = Zero_Label.copy()
        Label_Lat[2] = float(1)
        Label_Lat[6] = float(1)
        Label_Long = Zero_Label.copy()
        Label_Long[2] = float(1)
        Label_Long[7] = float(1)
        
        Label_Noise = Zero_Label.copy()
        Label_Noise[1] = float(1)
        
        add = 0
        if num_labels == 11:
            add = 5
        Label_Grad2 = Zero_Label.copy()
        Label_Grad2[2] = float(1)
        Label_Grad2[3+add] = float(1)
        Label_Min2 = Zero_Label.copy()
        Label_Min2[2] = float(1)
        Label_Min2[4+add] = float(1)
        Label_Sek2 = Zero_Label.copy()
        Label_Sek2[2] = float(1)
        Label_Sek2[5+add] = float(1)
        All_Labels_8 = [Label_Grad1, Label_Min1, Label_Sek1, Label_Lat, Label_Grad2, Label_Min2, Label_Sek2, Label_Long]
        All_Labels_6 = [Label_Grad1, Label_Min1, Label_Lat, Label_Grad2, Label_Min2, Label_Long]
        Labels = []
        for i in TCord:
            Labels.append(False)
        for i in range(len(CordL)):
            Cur_Tokens = Tokenizer.tokenize(CordL[i])
            Cur_Tokens_Following = Cur_Tokens.copy()
            Cur_Tokens_Following[0] = "##" + Cur_Tokens_Following[0]
            for j in range(len(TCord)-len(Cur_Tokens)+1):
                if TCord[j:j+len(Cur_Tokens)] == Cur_Tokens or TCord[j:j+len(Cur_Tokens)] == Cur_Tokens_Following:
                    for k in range(len(Cur_Tokens)):
                        if not Labels[j+k]:
                            if len(CordL) == 8:
                                Labels[j+k] = All_Labels_8[i]
                            else:
                                Labels[j+k] = All_Labels_6[i]
        for i in range(len(Labels)):
            if not Labels[i]:
                Labels[i] = Label_Noise

        return Labels





    def Replace(ParCord):
        Irrel_Label = []
        for i in range(num_labels):
            Irrel_Label.append(float(0))
        Irrel_Label[0] = float(1)
        (Par, ListOfCoords) = ParCord
        NSLIST = ["N", "S"]
        WELIST = ["W", "E"]
        
        Coords_In_This_Par = []

        while Coords_In_This_Par == []: # If the change of coordinates results in something the regex doesnt find
            ParCopy = Par
            for (Cord, StrCord) in ListOfCoords:
                random.shuffle(All_Coordinates)
                (NewCords, StrNewCords) = All_Coordinates[0]
                for i in range(len(NewCords)):
                    if NewCords[i].isnumeric():
                        StrNewCords = StrNewCords.replace(NewCords[i], str(random.randint(0, 90)))
                    else:
                        if NewCords[i] in NSLIST:
                            StrNewCords = StrNewCords.replace(NewCords[i], random.choice(NSLIST))
                        else:
                            StrNewCords = StrNewCords.replace(NewCords[i], random.choice(WELIST))

                ParCopy = ParCopy.replace(StrCord, StrNewCords)
            (Six, Eight, NF, E) = mc.find_coordinates(ParCopy)
            Found_Coords = Six + Eight
            for (Coordlist, StringCoord, Pppar) in Found_Coords:
                Coords_In_This_Par.append((Coordlist, StringCoord))

        Par = ParCopy
        # Par
        # Coords_In_This_Par
        # Now: Labels
        TokenPar = Tokenizer.tokenize(Par)
        TokenCords = []
        for (CLIST, SCORD) in Coords_In_This_Par:
            TokenCords.append((Tokenizer.tokenize(SCORD), CLIST, SCORD))
            
        FullLabels = []
        for i in TokenPar:
            FullLabels.append(False)
        for (TCord, CordL, SCord) in TokenCords:
            CordLabels = get_Labels_for_coords(TCord, CordL, SCord)                      
            LenOfCords = len(TCord)
            for i in range(len(TokenPar)-LenOfCords):
                if TokenPar[i:i+LenOfCords] == TCord:
                    for j in range(LenOfCords):
                        if not FullLabels[i+j]:
                            FullLabels[i+j] = CordLabels[j]
        for i in range(len(FullLabels)):
            if not FullLabels[i]:
                FullLabels[i] = Irrel_Label

        return(Par, FullLabels, Coords_In_This_Par)

    class Dataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            global Storage
            Irrel_Label = []
            Padded_Label = []
             # Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
            for i in range(num_labels):
                Irrel_Label.append(float(0))
                Padded_Label.append(float(0))
 # Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long

            Irrel_Label[0] = float(1)
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
                            NoisePar = NoisePar.replace(StringC, newcs)
                        NoisePar = NoisePar
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
                        if Labels[i][2] == float(1):
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
            Attention_Mask = []
            for i in range(len(Labels)):
                Attention_Mask.append(1)
            for i in range(PadLength-len(Labels)):
                TSPcoded.append(0)
                Labels.append(Padded_Label.copy())
                Attention_Mask.append(0)
 
            item = {}
            item['input_ids'] =  torch.tensor(TSPcoded)
            item['labels'] = torch.tensor(Labels)
            item['attention_mask'] = torch.tensor(Attention_Mask)

            return item

        def __len__(self):
            return DatasetLength
    
    import time
    TrainData = Dataset()
    Training_Loader = DataLoader(TrainData, batch_size = Batch_Size_Train)
    BCEWLL = torch.nn.BCEWithLogitsLoss(pos_weight = Pos_Weight_Vector).to(device)
    Loss_History = []
    Counter = 0
    Strange_Happenings = []
    Time_For_Batch = []
    Starttime = time.time()
    PassedSeconds = 0
    while time.time() - Starttime < Stoptime:
        for batch in Training_Loader:
            sstime = time.time()
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            att_mask = batch['attention_mask'].to(device)
            Output = Model(input_ids, attention_mask = att_mask)
            Logits = Output.logits
            Loss = BCEWLL(Logits, labels)
            lossnum = Loss.item()
            if Loss_History:
                if lossnum > Loss_History[-1]*100:
                    print("Jump in loss detected:")

                    Errordict = {}
                    Errordict["Step"] = Counter
                    Errordict["Old_Loss"] = Loss_History[-1]
                    Errordict["New_Loss"] = lossnum
                    Errordict["Factor"] = lossnum/Loss_History[-1]
                    Errordict["Paragraph_ids"] = input_ids
                    Errordict["Labels"] = labels
                    Errordict["Attention_Mask"] = att_mask
                    Errordict["Paragraph_tokens_and_labels_and_att"] = []
                    print(str(Loss_History[-1]) + " to " + str(lossnum) + "(factor " + str(Errordict["Factor"]) + ") in step " + str(Counter))
                    for jj in range(Batch_Size_Train):
                        Errordict["Paragraph_tokens_and_labels_and_att"].append((Tokenizer.convert_ids_to_tokens(input_ids[jj]), labels[jj], att_mask[jj]))
                    Errordict["Output"] = Output
                    Strange_Happenings.append(Errordict)
            Loss_History.append(lossnum)
            Loss.backward()
            optim.step()
            PassedSeconds += time.time() - Starttime
            if PassedSeconds > 600:
                print(Code + "_" + Clarifier + " with loss of " + str(round(lossnum, 6)) + " (" + str(Counter) + " steps, " + str(round(time.time() - Starttime, 2)) + "/" + str(Stoptime) + " seconds)")
                PassedSeconds = 0
            Counter += 1
            eetime = time.time()
            Time_For_Batch.append(eetime-sstime)
    endtime = time.time()
    global Storage
    Storage = False
    FullTime = endtime - Starttime

    mdl = "TC6_TT_" + Clarifier + "_" + Code
    ModName = mdl + "_Model/"
    Model.save_pretrained(CurDir + "/Models/" + ModName)

    import pickle
    HistoryOutputPlace = CurDir + "/Results/TC6_TT_Loss/"
    ErrorOutputPlace = CurDir + "/Results/TC6_TT_Errors/" 
    if not os.path.isdir(HistoryOutputPlace):
        os.mkdir(HistoryOutputPlace)
    if not os.path.isdir(ErrorOutputPlace):
        os.mkdir(ErrorOutputPlace)
    with open(HistoryOutputPlace + Code + "_" + Clarifier + ".pickle", "wb") as file:
        pickle.dump(Loss_History, file)
    with open(ErrorOutputPlace + Code + "_" + Clarifier + ".pickle", "wb") as file:
        pickle.dump(Strange_Happenings, file)
    Parameters["FullTime"] = FullTime
    Parameters["Pos_Weight_Vector"] = Pos_Weight_Vector
    Parameters["Len_los_history"] = len(Loss_History)
    Parameters["Time_Per_Batch"] = Time_For_Batch    
    with open(CurDir + "/Models/" + ModName + "/" + "Parameters.pickle", "wb") as file:
        pickle.dump(Parameters, file)
    print("Model " + mdl + " saved.")
