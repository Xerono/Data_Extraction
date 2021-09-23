ListOfRelevants = []
ListOfRelevants.append(("0011", False, 0.99))
ListOfRelevants.append(("0110", False, 0.99))
ListOfRelevants.append(("1010", False, 0.99))
ListOfRelevants.append(("0010", False, 0.99))
ListOfRelevants.append(("0110", False, 0.95))

ListOfRelevants.append(("0110", True, 0.95))
ListOfRelevants.append(("1010", True, 0.95))



import os
import sqlite3
import Module_Coordinates as mc
CurDir = os.getcwd()
Database = CurDir + "/Files/Database.db"
Con = sqlite3.connect(Database)
Cur = Con.cursor()
MaxLength = 500 # Tokens
xs = "Select * FROM Pars"
OriginalPars = Cur.execute(xs).fetchall()
Con.close()

from transformers import BertTokenizerFast
TokenizerC = BertTokenizerFast.from_pretrained(os.getcwd() + "/Custom_Tokenizer/")
TokenizerN = BertTokenizerFast.from_pretrained("bert-base-cased")
PwC = []
for (FPID, File, Par) in OriginalPars:
    (Six, Eight, NF, E) = mc.find_coordinates(Par)
    Found_Coords = Six + Eight
    Coords = []
    if len(Found_Coords)>0 and len(TokenizerC.tokenize(Par))<=MaxLength and len(TokenizerN.tokenize(Par))<=MaxLength:
        PwC.append((FPID, File, Par))

import random
random.shuffle(PwC)

Split = 80



Trainingdata = PwC[:int(len(PwC)/100*Split)]
Testdata = PwC[int(len(PwC)/100*Split):]

import pickle
with open(CurDir + "/Files/TC6_TT_Training.pickle", "wb") as file:
    pickle.dump(Trainingdata, file)
with open(CurDir + "/Files/TC6_TT_Test.pickle", "wb") as file:
    pickle.dump(Testdata, file)

print("Created test and training")




ListOfModels = []
for (Paras, Cust_Tok, Treshold) in ListOfRelevants:
    if (Paras, Cust_Tok) not in ListOfModels:
        ListOfModels.append((Paras, Cust_Tok))




import Create_TC6_TT_Model as NewTC6Model
for (Paras, Cust_Tok) in ListOfModels:
    print("Starting:")
    print(Paras)
    print(Cust_Tok)
    NewTC6Model.create(Paras, Cust_Tok)

print("Finished model creation")
TorT = ["Training", "Test", "All"]

import Use_TC6_TT_Model as UseTC6Model


for (Paras, Cust_Tok, Treshold) in ListOfRelevants:
    for Training in TorT:
        UseTC6Model.use(Paras, Cust_Tok, Treshold, Training)
