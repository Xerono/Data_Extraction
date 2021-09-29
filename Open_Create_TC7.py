import sys
if len(sys.argv)==2:
    This_Batch = int(sys.argv[1])
else:
    print("No batch specified")
    sys.exit()


Vars = ["0", "1"]
Batch1 = []
Batch2 = []
Batch3 = []
Batch4 = []
Batches = [Batch1, Batch2, Batch3, Batch4]
Ends = []
Fronts = []
for i in Vars:
    for j in Vars:
            Ends.append(i + j)
            Fronts.append(i + j)
i = 0
for front in Fronts:
    for end in Ends:
        Batches[i].append(front + end)
    i+=1

import Create_TC6_Model as NewTC6Model
ThisBatch = Batches[This_Batch]

import os
import sqlite3
import Module_Coordinates as mc
CurDir = os.getcwd()
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
MaxLength = 480 # Tokens
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

from transformers import BertTokenizerFast
TokenizerC = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
TokenizerN = BertTokenizerFast.from_pretrained("bert-base-cased")
PwC = []
Letterdict = {}
for (FPID, File, Par) in OriginalPars:
    (Six, Eight, NF, E) = mc.find_coordinates(Par)
    Found_Coords = Six + Eight
    Coords = []
    if len(Found_Coords)>0 and len(TokenizerC.tokenize(Par))<=MaxLength and len(TokenizerN.tokenize(Par))<=MaxLength:
        PwC.append((FPID, File, Par))

import random
random.shuffle(PwC)

Nonletters = [" ", ",", ";"]
for (FPID, File, Par) in PwC:
    (S, E, NF, EE) = mc.find_coordinates(Par)
    fc = S + E
    for ((sc, c, p)) in fc:
        ccopy = c
        for stri in sc:
            ccopy = ccopy.replace(stri, "", 1)
        for letter in ccopy:
            if letter not in Nonletters:
                if letter not in Letterdict.keys():
                    Letterdict[letter] = 1
                else:
                    Letterdict[letter] += 1
AllNoise = 0
for k in Letterdict.keys():
    AllNoise += Letterdict[k]
Chances = {}
for k in Letterdict.keys():
    Chances[k] = Letterdict[k]/AllNoise

Koordinatenformen = []
for (FPID, File, Par) in PwC:
    (S, E, NF, EE) = mc.find_coordinates(Par)
    fc = S + E
    for ((sc, c, p)) in fc:
        ccopy = c
        for stri in sc:
            ccopy = ccopy.replace(stri, "", 1)
        for lett in Nonletters:
            ccopy = ccopy.replace(lett, "")
        Chance = 1
        for lette in ccopy:
            if lette in Chances.keys():
                Chance = Chance*Chances[lette]
        Koordinatenformen.append((c, sc, Chance))
Seen_Chanc = {}
Koordinatenformen.sort(key=lambda x: x[2])
for ((c, sc, ccc)) in Koordinatenformen:
    if ccc not in Seen_Chanc.keys():
        Seen_Chanc[ccc] = [(sc, c)]
    else:
        Seen_Chanc[ccc].append((sc, c))
op = []
for k in Seen_Chanc.keys():
    op.append(Seen_Chanc[k][0])

CurDir = os.getcwd()
if not os.path.isfile(CurDir + "/Files/TC7_Coords.pickle"):
    with open(CurDir + "/Files/TC7_Coords.pickle", "wb") as fff:
        pickle.dump(op)
Tresholds = [float(0.4), float(0.5), float(0.6), float(0.7), float(0.8), float(0.9), float(0.95), float(0.99)]

import Create_TC7_Model as NewTC7Model
import Use_TC7_Model as UseTC7Model
for Paras in ThisBatch:
    for cust_Tok in [True, False]:
        NewTC7Model.create(tuple(Paras), cust_Tok)
        for trsh in Tresholds:
            UseTC7Model.use(tuple(Paras), cust_Tok, trsh)
