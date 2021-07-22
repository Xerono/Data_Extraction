# Initialisierung

from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
import torch

Basemodel = "distilbert-base-uncased" # BERT müsste erst heruntergeladen werden, beim Anwenden wird das auf BERT laufen (darum auch die Warnung)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LabelCount = 8 # Acht mögliche Label pro Token

Tokenizer = BertTokenizerFast.from_pretrained(Basemodel)
Model = BertForTokenClassification.from_pretrained(Basemodel, num_labels=LabelCount).to(device)


TS = "Test" # Ersetzt hier einen kompletten Paragraphen, um die Übersichtlichkeit zu wahren

# Anwendung des Models
StrEnc = Tokenizer(TS, return_tensors="pt").to(device)
Output = Model(**StrEnc)


# Loss-Funktion wird initialisiert, Gewicht für jedes Label ist gleich
BCEWLL = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones([LabelCount])).to(device)

# Entferne CLS und SEP aus den Ergebnissen, da nicht relevant
Logits_Without_CLSEP = Output.logits[-1][1:-1]

# (Im Testbeispiel zufällige) Labels für jeden Token der Eingabe (in diesem Fall nur 1)
import random
Labels = []
for i in range(LabelCount):
    j = random.choice([0, 1])
    Labels.append(float(j)) # Ohne Float funktioniert es nicht
    
ListOfAllLabelsForInput = [Labels]

ClassLabels = torch.tensor(ListOfAllLabelsForInput).to(device)

# Wende Funktion auf ausgegebene Logits und erzeugte Label für Eingabestring an
# Sigmoidfunktion ist laut Docs bereits integriert
Loss = BCEWLL(Logits_Without_CLSEP, ClassLabels)

print(Loss.item())

# Hier würde nun Loss.backward() und optim.step() folgen, analog zur bisherigen CrossEntropy-Variante


