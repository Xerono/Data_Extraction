import sys
if len(sys.argv)==2:
    This_Batch = int(sys.argv[1])
else:
    print("Missing parameter")
    sys.exit()


import os


Batch1 = []
Batch2 = []
Batch3 = []
Batch4 = []
Vars = ["0", "1"]
Batches = [Batch1, Batch2, Batch3, Batch4]
Ends = []
for i in Vars:
    for j in Vars:
        for k in Vars:
            Ends.append(i + j + k)
Fronts = []
for i in Vars:
    for j in Vars:
        Fronts.append(i + j)
alls = []
for i in range(4):
    for end in Ends:
        Batches[i].append(Fronts[i] + end)

import Create_TC5_Model as NewTC5Model
ThisBatch = Batches[This_Batch]
import subprocess
for Paras in ThisBatch:
    NewTC5Model.create(tuple(Paras))

print("Finished batch " + str(This_Batch))
