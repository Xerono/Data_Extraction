def create(Inputs):
    (Cutting_Pars, CoordsToNoise, Delete_Coords, Detailed_Labels) = Inputs
    Cut_Par = bool(int(Cutting_Pars))
    Coord_To_Noise = bool(int(CoordsToNoise))
    Delete_Teilcoords = bool(int(Delete_Coords))
    Detailed_Labels = bool(int(Detailed_Labels))

    
    Basemodel = "bert-base-cased"
    Basemodel = "distilbert-base-uncased"

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
    with open(CurDir + "/Files/TC3_Training.pickle", "wb") as file:
        pickle.dump(Trainingdd, file)
    with open(CurDir + "/Files/TC3_Test.pickle", "wb") as file:
        pickle.dump(Testdd, file)
    PwC = Trainingdd



    import torch


    from transformers import BertTokenizerFast
    from transformers import BertForTokenClassification


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
        if Detailed_Labels: # Padded, Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
            for i in range(12):
                Basic_Label.append(0)
        else: # Padded, Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long
            for i in range(9):
                Basic_Label.append(0)

        Noise = ""
        Labels = []
        for i in range(random.choice([1,2])):
            Noise = Noise + random.choice(Symbols)
            CurLabel = Basic_Label.copy()
            CurLabel[2] = 1
            Labels.append(CurLabel)
            
        return (Noise, Labels)

    def generate_coords():
       
        PotCoords = []
        Labels = []
        CoordsString = ""
        Basic_Label = []
        EightCoords = random.choice([True, False])
        
        if Detailed_Labels: # Padded, Irrelevant, Noise, Coord, Grad1, Min1, Sek1, Lat, Long, Grad2, Min2, Sek2
            for i in range(12):
                Basic_Label.append(0)
        else: # Padded, Irrelevant, Noise, Coord, Grad, Min, Sek, Lat, Long
            for i in range(9):
                Basic_Label.append(0)
                

        Grad1 = str(random.randint(0, 90))
        for i in range(len(Grad1)):
            Cur = Basic_Label.copy()
            Cur[3] = 1
            Cur[4] = 1
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Grad1 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)

        Min1 = str(random.randint(0, 60))
        for i in range(len(Min1)):
            Cur = Basic_Label.copy()
            Cur[3] = 1
            Cur[5] = 1
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Min1 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)
            
        if EightCoords:
            Sek1 = str(random.randint(0, 60))
            for i in range(len(Sek1)):
                Cur = Basic_Label.copy()
                Cur[3] = 1
                Cur[6] = 1
                Labels.append(Cur)
            (Noise, NLabels) = generate_noise()
            CoordsString = CoordsString + Sek1 + Noise
            for NoiseLabels in NLabels:
                Labels.append(NoiseLabels)

        Lat = random.choice(["N", "S"])
        Cur = Basic_Label.copy()
        Cur[3] = 1
        Cur[7] = 1
        Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Lat + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)

        Grad2 = str(random.randint(0, 90))
        for i in range(len(Grad2)):
            Cur = Basic_Label.copy()
            Cur[3] = 1
            if Detailed_Labels:
                Cur[9] = 1
            else:
                Cur[4] = 1
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Grad2 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)
            
        Min2 = str(random.randint(0, 60))
        for i in range(len(Min2)):
            Cur = Basic_Label.copy()
            Cur[3] = 1
            if Detailed_Labels:
                Cur[10] = 1
            else:
                Cur[5] = 1
            Labels.append(Cur)
        (Noise, NLabels) = generate_noise()
        CoordsString = CoordsString + Min2 + Noise
        for NoiseLabels in NLabels:
            Labels.append(NoiseLabels)
            
        if EightCoords:
            Sek2 = str(random.randint(0, 60))
            for i in range(len(Sek2)):
                Cur = Basic_Label.copy()
                Cur[3] = 1
                if Detailed_Labels:
                    Cur[11] = 1
                else:
                    Cur[6] = 1
                Labels.append(Cur)
            (Noise, NLabels) = generate_noise()
            CoordsString = CoordsString + Sek2 + Noise
            for NoiseLabels in NLabels:
                Labels.append(NoiseLabels)

        Lon = random.choice(["W", "E"])
        Cur = Basic_Label.copy()
        Cur[3] = 1
        Cur[8] = 1
        Labels.append(Cur)
        CoordsString = CoordsString + Lon

        if EightCoords:
            PotCoords = (Grad1, Min1, Sek1, Lat, Grad2, Min2, Sek2, Lon)
        else:
            PotCoords = (Grad1, Min1, Lat, Grad2, Min2, Lon)
        return(PotCoords, CoordsString, Labels)

            
    
        
    def Replace(ParCord):
        (Par, ListOfCoords) = ParCord
        FullNewCoords = []
        for (Coord, String) in ListOfCoords:
            (PotCoords, CoordsString, Labels) = generate_coords()
            Par = Par.replace(String, CoordsString)
            FullNewCoords.append((PotCoords, CoordsString, Labels))
            


