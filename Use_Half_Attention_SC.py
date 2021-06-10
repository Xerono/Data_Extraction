#Variables

Soils_Q = True # If false, Coordinate Model will be used
Layernums = [0] # Layers from which attention should be used

Debug_Q = False # Output debug information






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
    # Attention for CLS per head per layer
    Layer = {}
    for layernum in range(len(Add_Dict)):
        Layer[layernum] = {}
        for headnum in range(len(Add_Dict[0])):
            Layer[layernum][headnum] = []
            for tokennum in range(len(Add_Dict[0][0])):
                Layer[layernum][headnum].append(Add_Dict[layernum][headnum][tokennum][0]) # cls
    return Layer

def get_sum_of_heads(cls):
    # Sum Of heads per layer
    soh = {}
    for layernum in range(len(cls)):
        soh[layernum] = {}
        for tokennum in range(len(cls[0][0])):
            tokensum = 0
            for headnum in range(len(cls[0])):
                tokensum += cls[layernum][headnum][tokennum]
            soh[layernum][tokennum] = tokensum
    return soh

def check_word(Word, ID, Tokens):
    # Find words to token
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
                    if ID+j>=len(Tokens):
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
    
    
def most_important_words(soh, Tokens):
    attentionsumlayer = {}
    for tokennum in soh[0].keys():
        attentionsumlayer[tokennum] = 0
    for layer in soh.keys():
        if layer in Layernums: # Only attention from predefined layers
            curlist = list(soh[layer].values())
            for tokennum in range(len(curlist)):
                attentionsumlayer[tokennum] += curlist[tokennum]
    TokenAttentionList = []
    for tokennum in attentionsumlayer.keys():
        if tokennum != 0 and tokennum != len(Tokens)-1: # remove cls/sep
            TokenAttentionList.append((attentionsumlayer[tokennum], Tokens[tokennum]))
    return TokenAttentionList

def get_attention_words(Model, Tokenizer, String): 
    adddict, tokens = get_attn_tokens(Model, Tokenizer, String)
    cls = get_cls_atts(adddict)
    soh = get_sum_of_heads(cls)
    miw = most_important_words(soh, tokens)
    return miw

def comp_splits(A, B):
    sumA = 0
    sumB = 0
    for (att, token) in A:
        sumA += att
    for (att, token) in B:
        sumB += att
    return sumA-sumB

def halfatt(important_words):
    Compare_Splits = {}
    for i in range(len(important_words)):
        SplitA = important_words[:i]
        SplitB = important_words[i:]
        Compare_Splits[i] = comp_splits(SplitA, SplitB)
    return Compare_Splits

def stitch_together(ListOfStrings):
    Orig = ""
    for Str in ListOfStrings:
        Orig = Orig + Str + " "
    return Orig[:-1]


def find_splitindex(Str, Model, Tokenizer):
    att_word = get_attention_words(Model, Tokenizer, Str)
    tokens = []
    atts = []
    for (att, word) in att_word:
        atts.append(att)
        tokens.append(word)
    halfatts = []
    for i in range(len(atts)):
        halfatts.append((atts[i], tokens[i]))
    pot_Comp_Splits = halfatt(halfatts)
    Comp_Splits = list(map(abs, list(pot_Comp_Splits.values())))
    minattdifference = min(Comp_Splits)
    minentry = Comp_Splits.index(minattdifference)
    return minentry, tokens

def split_check(Str, Model, Tokenizer, Perc):
    minentry, tokens = find_splitindex(Str, Model, Tokenizer)
    splittoken = tokens[minentry]
    splitword = check_word(splittoken, minentry, tokens)
    Original_Seperated = Str.split(" ")
    if splitword not in Original_Seperated:
        splitposition = int(len(Original_Seperated)/2)
    else:
        splitposition = Original_Seperated.index(splitword)
        if splitposition == 0:
            splitposition = 1
    Part1_cut = Original_Seperated[:splitposition]
    Part2_cut = Original_Seperated[splitposition:]
    Part1 = stitch_together(Part1_cut)
    Part2 = stitch_together(Part2_cut)
    if Debug_Q:
        print(Str)
        print("-------------")
        print(splitword)
        print("----")
        print(Part1)
        print("----")
        print(Part2)
        print("---------------------")
    
        input()
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



def get_label(Str, Model, Tokenizer):
    StrEnc = Tokenizer(Str, return_tensors='pt').to(device)
    Output = Model(**StrEnc)
    Softmaxed = Output.logits.softmax(1)
    Label = Softmaxed[0][1].item()
    return Label

if not Soils_Q:
    import re
    def coord_regex(Stringlist):
        Found = False
        # Find xx°xx'N
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

CurDir = os.getcwd()





import sqlite3

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



Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()






Paragraphs = {}
Type = "Half_Attention"
Layers = ""
for layer in Layernums:
    Layers = Layers + "_L" + str(layer) 

if Soils_Q:
    Model_path = CurDir + "/Models/SC_Model_Soils/"
    Model_Name = Type + Layers + "_Soils"
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
else:
    Model_path = CurDir + "/Models/SC_Model_Coordinates/"
    Model_Name = Type + Layers + "_Coords"
    PercentageO = 70

    xs = "Select FPID, NumOfCoords from Coordinates"
    CoordsPars = Cur.execute(xs).fetchall()
    xs = "Select * FROM Pars"
    Pars = Cur.execute(xs).fetchall()
    Con.close()
    
    for (ParID, NumOfCoords) in CoordsPars:
        Paragraphs[ParID] = NumOfCoords



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

   
import torch
import transformers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Model = transformers.DistilBertForSequenceClassification.from_pretrained(Model_path, output_attentions = True)
Model.eval()
Model.to(device)

from transformers import DistilBertTokenizerFast
PreTrainedModel = "distilbert-base-uncased"
Tokenizer = DistilBertTokenizerFast.from_pretrained(PreTrainedModel)


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
    enctext = Tokenizer(Par, return_tensors='pt').to(device)
    CalcLabel = get_label(Par, Model, Tokenizer)
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
        imp_words_checked = split_check(Par, Model, Tokenizer, Percentage)
        if Soils_Q:
            for (Soil, SoilD) in Paragraphs[FPID]:
                Found = False
                for word in imp_words_checked:  
                    if (Soil in word) or (SoilD in word) or word in Soil or word in SoilD:
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
        else:
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
        print(str(Status) + " von " + str(len(Dataset)))

results_list = []

results_list.append((Model_Name, Resultsdict[11], Resultsdict[10],
                     Resultsdict[o1], Resultsdict[o0], Resultsdict[B1F],
                     Resultsdict[B1N], Resultsdict[B0F], Resultsdict[B0N], PercentageO))

Database = CurDir + "/Results/Results.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
sql_command = "INSERT INTO Results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
Cur.executemany(sql_command, results_list)
Con.commit()
Con.close()
print("Finished")
