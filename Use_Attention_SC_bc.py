# Variables


NumOfBest = 10 # Length of list of words with the highest attention

modeltypes = ["Soils_bc", "Soilless_bc", "Coordinates_bc"]




for ModelType in modeltypes:
    import os

    CurDir = os.getcwd()

    ModPath = CurDir + "/Models/"


    import sqlite3
    Database = CurDir + "/Files/Database.db"
    Con = sqlite3.connect(Database)
    Cur = Con.cursor()
    Paragraphs = {}

    if ModelType == "Soils_bc" or ModelType == "Soilless_bc":

        xs = "Select FPID, Soil, SoilD FROM Soils"
        Pars_Type = Cur.execute(xs).fetchall()
        xs = "Select * FROM Pars"
        Pars = Cur.execute(xs).fetchall()
        for (ParID, Soil, SoilD) in Pars_Type:
            if ParID not in Paragraphs.keys():
                Paragraphs[ParID] = [(Soil, SoilD)]
            else:
                Paragraphs[ParID].append((Soil, SoilD))

        PercentageO = 90

    if ModelType == "Coordinates_bc":
        xs = "Select FPID, NumOfCoords FROM Coordinates"
        CoordPars = Cur.execute(xs).fetchall()
        xs = "Select * FROM Pars"
        Pars = Cur.execute(xs).fetchall()

        for (ParID, NumOfCoords) in CoordPars:
            Paragraphs[ParID] = NumOfCoords

        PercentageO = 70

    Model_Path = ModPath + "SC_Model_" + ModelType + "/"
    Con.close()




    def create_Attdict(attention):
        AttentionDict = {}
        layernum = 0
        headnum = 0
        for layer in attention: #6
            AttentionDict[layernum] = {}
            for head in layer[0]: #12
                AttentionDict[layernum][headnum] = head.tolist() #[numtokens*[numtokens]]
                headnum += 1
            layernum += 1
            headnum = 0
        return AttentionDict

    def get_attn_tokens(Model, Tokenizer, String):
        inputs = Tokenizer.encode_plus(String, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids'].to(device)
        attention = Model(input_ids)[-1]
        input_id_list = input_ids[0].tolist()
        Tokens = Tokenizer.convert_ids_to_tokens(input_id_list)
        AddDict = create_Attdict(attention)
        return(AddDict, Tokens)

    def get_cls_atts(Add_Dict):
        Layer = {}
        for layernum in range(len(Add_Dict)):
            Layer[layernum] = {}
            for headnum in range(len(Add_Dict[0])):
                Layer[layernum][headnum] = []
                for tokennum in range(len(Add_Dict[0][0])):
                    Layer[layernum][headnum].append(Add_Dict[layernum][headnum][tokennum][0]) # cls
        return Layer

    def get_sum_of_heads(cls):
        soh = {}
        # Annahme: Summe über Attentions verschiedener Heads pro Token macht Sinn
        for layernum in range(len(cls)):
            soh[layernum] = {}
            for tokennum in range(len(cls[0][0])):
                tokensum = 0
                for headnum in range(len(cls[0])):
                    tokensum += cls[layernum][headnum][tokennum]
                soh[layernum][tokennum] = tokensum
        return soh

    def check_word(Word, ID, Tokens):
        plc = "##"
        if ID >= len(Tokens)-1:
            if plc in Word:
                j = 1
                Word = Word.replace(plc, "")
                while plc in Tokens[ID-1]:
                    Word = Tokens[ID-j].replace(plc, "") + Word
                    j += 1
                    if ID-j<0:
                        break
        else:
            if plc not in Word and plc in Tokens[ID+1]: # >te< ##st ##word testword testword
                j = 1
                while plc in Tokens[ID+j]:
                    Word = Word + Tokens[ID+j].replace(plc, "")
                    j += 1
                    if ID+j==len(Tokens):
                        break
            elif plc not in Word and plc not in Tokens[ID+1]: # te ##st ##word >testword< testword
                Word = Word
            elif plc in Word and plc not in Tokens[ID+1]: # te ##st >##word< testword testword
                j = 1
                Word = Word.replace(plc, "")
                while plc in Tokens[ID-j]:
                    Word = Tokens[ID-j].replace(plc, "") + Word
                    j += 1
                    if ID-j<0:
                        break
                Word = Tokens[ID-j] + Word
            elif plc in Word and plc in Tokens[ID+1]: # te >##st< ##word testword testword
                    j = 1
                    while plc in Tokens[ID+j]:
                        Word = Word + Tokens[ID+j].replace(plc, "")
                        j += 1
                        if ID+j>len(Tokens):
                            break
                    j = 1
                    Word = Word.replace(plc, "")
                    while plc in Tokens[ID-j]:
                        Word = Tokens[ID-j].replace(plc, "") + Word
                        j += 1
                        if ID-j<0:
                            break
                    Word = Tokens[ID-j].replace(plc, "") + Word
        return Word
        
        
    def most_important_words(soh, Tokens, NumOfBest):
        import numpy as np
        BestWords = {}
        for layernum in range(len(soh)):
            BestWords[layernum] = []
            cursoh = list(soh[layernum].values())
            for i in range(NumOfBest):
                curmax = np.argmax(cursoh)
                del cursoh[curmax]
                word = check_word(Tokens[curmax], curmax, Tokens)
                BestWords[layernum].append(word)
        return BestWords

    def get_attention_words(Model, Tokenizer, String, NumOfBest): 
        adddict, tokens = get_attn_tokens(Model, Tokenizer, String)
        cls = get_cls_atts(adddict)
        soh = get_sum_of_heads(cls)
        if NumOfBest>len(tokens):
            NumOfBest = int(len(tokens)/2)
        miw = most_important_words(soh, tokens, NumOfBest)
        return miw

    def check_words(important_words):
        wordlist = []
        for dimension in important_words:
            for impword in important_words[dimension]:
                wordlist.append(impword)
        wordcounts = {i:wordlist.count(i) for i in wordlist}
        Maxlist = max(list(wordcounts.values()))
        checked_words = []
        for word in wordcounts.keys():
            if wordcounts[word] == Maxlist:
                checked_words.append(word)
            
        return checked_words


    if ModelType == "Coordinates_bc":
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







    import torch
    import transformers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model = transformers.BertForSequenceClassification.from_pretrained(Model_Path, output_attentions = True)
    Model.eval()
    Model.to(device)


    from transformers import BertTokenizerFast
    PreTrainedModel = 'bert-base-cased'
    Tokenizer = BertTokenizerFast.from_pretrained(PreTrainedModel)






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

    Models = [Model]
    Resultsdict = {}

    o1 = "NullEins"
    o0 = "NullNull"
    B1F = "B1SoilFound"
    B1N = "B1SoilNotF"
    B0F = "B0SoilFound"
    B0N = "B0SoilNotF"

    for Model in Models:
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
        enctext = Tokenizer(Par, return_tensors='pt').to(device)
        for Model in Models:
            Output = Model(**enctext)
            SoftMax = Output.logits.softmax(-1)
            CalcLabel = SoftMax[0][1].item()
            OverPerc = False
            if Label == 1:
                if CalcLabel > Percentage:
                    Resultsdict[11]+=1
                    OverPerc = True
                else:
                    Resultsdict[o1]+=1
            else:
                if CalcLabel > Percentage:
                    Resultsdict[10]+=1
                    OverPerc = True
                else:
                    Resultsdict[o0]+=1
            if Label == 1:
                important_words = get_attention_words(Model, Tokenizer, Par, NumOfBest)
                imp_words_checked = check_words(important_words)
                if ModelType == "Soils_bc" or ModelType == "Soilless_bc":
                    for (Soil, SoilD) in Paragraphs[FPID]:
                        Found = False
                        for word in imp_words_checked:  
                            if (Soil in word) or (SoilD in word) or word in SoilD or word in Soil:
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
                if ModelType == "Coordinates_bc":
                    for i in range(Paragraphs[FPID]):
                        Found = coord_regex(imp_words_checked)
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
            pass
            #print(str(Status) + " von " + str(len(Dataset)))

    results_list = []

    ModName = "Attention_" + ModelType

    results_list.append((ModName, Resultsdict[11], Resultsdict[10], Resultsdict[o1], Resultsdict[o0]
                         , Resultsdict[B1F], Resultsdict[B1N], Resultsdict[B0F], Resultsdict[B0N], PercentageO)) 

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
    print("Finished " + ModelType)
