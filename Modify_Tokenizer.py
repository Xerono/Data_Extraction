Debug = False

from transformers import BertTokenizerFast
Basemodel = "bert-base-cased"
OriginalTokenizer = BertTokenizerFast.from_pretrained(Basemodel)

import os
Filesfolder = os.getcwd() + "/Files/"
Tokenizersave = "Custom_Tokenizer/"
OriginalTokenizer.save_pretrained(Filesfolder + Tokenizersave)


import json
Origvocab = "vocab.txt"
OriginalVocabLocation = Filesfolder + Tokenizersave + Origvocab 
OriginalVocabFile = open(OriginalVocabLocation, "r", encoding = "UTF-8")

OriginalVocab = []
NewVocab = []
Discarded = []
for line in OriginalVocabFile:
    CanWeKeepThis = True
    line = line.strip() # \n
    OriginalVocab.append(line)
    
    if "[" not in line and "]" not in line: # Keep defaults
        if len(line)>1: # Keep single numbers
            if line.isnumeric(): # Remove numbers >= 2
                CanWeKeepThis = False
            else: # contains non-number chars
                if line[0:2] == "##":
                    FollowUp = line[2:]
                    if len(FollowUp) > 1: # Single numbers are fine
                        if FollowUp.isnumeric():
                            CanWeKeepThis = False
    if CanWeKeepThis:
        NewVocab.append(line)
    else:
        Discarded.append(line)
        NewVocab.append("")
OriginalVocabFile.close()

if Debug:
    for i in range(len(NewVocab)):
        print(str(i) + "  -  " + NewVocab[i] + "  -  " + OriginalVocab[i])
        if NewVocab[i] != OriginalVocab[i]:
            input()
        
from shutil import copyfile
copyfile(OriginalVocabLocation, Filesfolder + Tokenizersave + "old_vocab.txt")

with open(OriginalVocabLocation, "w", encoding = "UTF-8") as file:
    for word in NewVocab:
        file.write(word + "\n")


ID = 0
import json
with open(Filesfolder + Tokenizersave + "tokenizer.json", "r", encoding = "UTF-8") as file:
    Tokenizer = json.load(file)
    for item in Discarded:
        Current_Number = Tokenizer["model"]["vocab"].pop(item)
        Tokenizer["model"]["vocab"]["[cunused" + str(ID) + "]"] = Current_Number
        ID += 1
        
copyfile(Filesfolder + Tokenizersave + "tokenizer.json", Filesfolder + Tokenizersave + "old_tokenizer.json")

with open(Filesfolder + Tokenizersave + "tokenizer.json", "w", encoding = "UTF-8") as file:
    json.dump(Tokenizer, file)

NewTokenizer = BertTokenizerFast.from_pretrained(Filesfolder + Tokenizersave)
Test = "Das ist ein Testsatz. 123 ist eine Zahl, 1 2 3 4ab teilweise auch. 23 45"
print("Original: " + str(OriginalTokenizer.tokenize(Test)))
print("Modified: " + str(NewTokenizer.tokenize(Test)))
