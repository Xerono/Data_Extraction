import sys
if len(sys.argv)==2:
    This_Batch = int(sys.argv[1])
else:
    This_Batch = 0


import os


Batch1 = []
Batch2 = []
Batch3 = []

Vars = ["0", "1"]
Batches = [Batch1, Batch2, Batch3]
Ends = []
for i in Vars:
    for j in Vars:
            Ends.append(i + j)
Fronts = []
for i in Vars:
    for j in Vars:
        Fronts.append(i + j)
alls = []
for i in range(3):
    for end in Ends:
        Batches[i].append(Fronts[i] + end)

import Create_TC4_Model as NewTC4Model
ThisBatch = Batches[This_Batch]
import subprocess
for Paras in ThisBatch:
    NewTC4Model.create(tuple(Paras))

print("Finished batch " + str(This_Batch))
