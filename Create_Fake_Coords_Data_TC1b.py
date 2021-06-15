# Load original paragraphs
import os
import sqlite3
import random

realmodels = ["distilbert-base-uncased", "distilbert-base-cased", "bert-base-cased", "bert-base-uncased"]

for tknz in realmodels:

    if tknz == "distilbert-base-uncased":
        from transformers import DistilBertTokenizerFast
        PreTrainedModel = tknz
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
        fn = "du"
    if tknz == "distilbert-base-cased":
        from transformers import DistilBertTokenizerFast
        PreTrainedModel = tknz
        Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)
        fn = "dc"
    if tknz == "bert-base-cased":
        from transformers import BertTokenizerFast
        PreTrainedModel = tknz
        Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
        fn = "bc"    
    if tknz == "bert-base-uncased":
        from transformers import BertTokenizerFast
        PreTrainedModel = tknz
        Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)
        fn = "bu"
    CurDir = os.getcwd()
    Database = CurDir + "/Files/Database.db"
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()

    xs = "Select * FROM Pars"
    OriginalPars = Cur.execute(xs).fetchall()
    Con.close()

    # Find coordinates and beautify

    Symbols = ["•", "H", "V", "¢", ".", "j", "J", "°", ",", ";", "Њ", "Ј", "U",
               '"', "″", "'", "o", "@", "؇", "-", "¶", "(", ")", "Љ", "±",
               ":", "µ", "/",
               "8", "9"] # Found by trial & error
    Additional_Symbols = ["and"] # used by coordinates, not used in creation

    random.seed("FakeDataGenerationRandomSeed")

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

    ParDict = {}

    for (Extracted_Coords, Found_Regex, Par) in Sixers + Eighters:
        if Par in ParDict.keys():
            ParDict[Par].append((Extracted_Coords, Found_Regex))
        else:
            ParDict[Par] = [(Extracted_Coords, Found_Regex)]

    Pre_Words = []
    Post_Words = []

    for Par in ParDict.keys():
        Parts = []
        PreCoordinates = []
        PostCoordinates = []
        PreWords = []
        PostWords = []
        for (ExCoords, FouRegex) in ParDict[Par]:
            Parts.append(Par.split(FouRegex))
        for (p1, p2) in Parts:
            for (ExCoords, FouRegex) in ParDict[Par]:
                p1 = p1.replace(FouRegex, "")
                p2 = p2.replace(FouRegex, "")
            PreCoordinates.append(p1.split(" "))
            PostCoordinates.append(p2.split(" "))
        for WordList in PreCoordinates:
            for Word in WordList:
                PreWords.append(Word)
        for WordList in PostCoordinates:
            for Word in WordList:
                PostWords.append(Word)
        PreWords = list(set(PreWords))
        PostWords = list(set(PostWords))
        for word in PreWords:
            if len(word)>0:
                Pre_Words.append(word)
        for word in PostWords:
            if len(word)>0:
                Post_Words.append(word)
    PreWords = list(set(Pre_Words))
    PostWords = list(set(Post_Words))



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
            
        

    def generate_six_coords():
        gradN = str(random.randint(0, 90))
        minN = str(random.randint(0, 59))
        NS = random.choice(["N", "S"])
        gradW = str(random.randint(0, 90))
        minW = str(random.randint(0, 59))
        WE = random.choice(["W", "E"])
        Labels = []
        Coords = []
        if len(gradN) == 1:
            Labels.append("Grad")
            Coords.append(gradN)
        else:
            Labels.append("Grad")
            Labels.append("Grad")
            Coords.append(gradN[0])
            Coords.append(gradN[1])

        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        if len(minN) == 1:
            Labels.append("Min")
            Coords.append(minN)
        else:
            Labels.append("Min")
            Labels.append("Min")
            Coords.append(minN[0])
            Coords.append(minN[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        Coords.append(NS)
        Labels.append("Latitude")
        
        
        if len(gradW) == 1:
            Labels.append("Grad")
            Coords.append(gradW)
        else:
            Labels.append("Grad")
            Labels.append("Grad")
            Coords.append(gradW[0])
            Coords.append(gradW[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        if len(minW) == 1:
            Labels.append("Min")
            Coords.append(minW)
        else:
            Labels.append("Min")
            Labels.append("Min")
            Coords.append(minW[0])
            Coords.append(minW[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        Coords.append(WE)
        Labels.append("Longitude")
        CoordsString = ""
        for i in Coords:
            CoordsString += i
        return (CoordsString, Labels)

    def generate_eight_coords():
        gradN = str(random.randint(0, 90))
        minN = str(random.randint(0, 59))
        sekN = str(random.randint(0, 59))
        NS = random.choice(["N", "S"])
        gradW = str(random.randint(0, 90))
        minW = str(random.randint(0, 59))
        sekW = str(random.randint(0, 59))
        WE = random.choice(["W", "E"])
        Labels = []
        Coords = []
        if len(gradN) == 1:
            Labels.append("Grad")
            Coords.append(gradN)
        else:
            Labels.append("Grad")
            Labels.append("Grad")
            Coords.append(gradN[0])
            Coords.append(gradN[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        if len(minN) == 1:
            Labels.append("Min")
            Coords.append(minN)
        else:
            Labels.append("Min")
            Labels.append("Min")
            Coords.append(minN[0])
            Coords.append(minN[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")

        if len(sekN) == 1:
            Labels.append("Sek")
            Coords.append(sekN)
        else:
            Labels.append("Sek")
            Labels.append("Sek")
            Coords.append(sekN[0])
            Coords.append(sekN[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        Coords.append(NS)
        Labels.append("Latitude")
        
        
        if len(gradW) == 1:
            Labels.append("Grad")
            Coords.append(gradW)
        else:
            Labels.append("Grad")
            Labels.append("Grad")
            Coords.append(gradW[0])
            Coords.append(gradW[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        if len(minW) == 1:
            Labels.append("Min")
            Coords.append(minW)
        else:
            Labels.append("Min")
            Labels.append("Min")
            Coords.append(minW[0])
            Coords.append(minW[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
        
        if len(sekW) == 1:
            Labels.append("Sek")
            Coords.append(sekW)
        else:
            Labels.append("Sek")
            Labels.append("Sek")
            Coords.append(sekW[0])
            Coords.append(sekW[1])
            
        for i in range(random.choice([1,2])):
            Coords.append(random.choice(Symbols))
            Labels.append("Noise")
            
        Coords.append(WE)
        Labels.append("Longitude")
        CoordsString = ""
        for i in Coords:
            CoordsString += i
        return (CoordsString, Labels)

    Maxlength = 917

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def create_paragraph():
        PrePar = []
        PostPar = []
        prewordslen = random.randint(5,50)
        postwordslen = random.randint(5, 50)
        for i in range(prewordslen):
            random.shuffle(PreWords)
            PrePar.append(PreWords[0])
        for i in range(postwordslen):
            random.shuffle(PostWords)
            PostPar.append(PostWords[0])
        if random.randint(0,1)==1:
            (Coords, Labels) = generate_eight_coords()
        else:
            (Coords, Labels) = generate_six_coords()
        PreString = ""
        PostString = ""
        for word in PrePar:
            PreString = PreString + word + " "
        for word in PostPar:
            PostString = PostString + word + " "
        PreString = PreString[0:-1]
        PostString = PostString[0:-1]
        TokenizedPre = Tokenizer.tokenize(split_string(PreString))
        TokenizedPost = Tokenizer.tokenize(split_string(PostString))
        PreLabels = []
        PostLabels = []
        for item in TokenizedPre:
            PreLabels.append("Nul")
        for item in TokenizedPost:
            PostLabels.append("Nul")
        FinalString = split_string(PreString) + " " + split_string(Coords) + " " + split_string(PostString)
        FinalLabels = PreLabels + Labels + PostLabels
        return(FinalString, FinalLabels)

    NumOfExamples = 100000
    Dataset = []
    Runner = 1
    import time
    starttime = time.time()
    while Runner <= NumOfExamples:
        if random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])>3:
            newpar, Labellist = create_paragraph()
        else:
            random.shuffle(NotFound)
            newpar = split_string(NotFound[0])
            splitted_ex = Tokenizer.tokenize(newpar)
            Labellist = []
            for i in range(len(splitted_ex)):
                Labellist.append("Nul")
        if len(newpar)<Maxlength:
            Dataset.append(((Runner, newpar), Labellist))
            Runner+=1
        if Runner % 1000 == 0:
            print(str(Runner) + "/" + str(NumOfExamples))
            print(time.time()-starttime)

    import pickle
    with open(CurDir + "/Files/FakeData_b_" + fn + ".pickle", 'wb') as handle:
        pickle.dump(Dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


