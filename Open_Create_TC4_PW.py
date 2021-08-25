import sys
if len(sys.argv)==2:
    This_Batch = int(sys.argv[1])
else:
    print("No batch specified")
    sys.exit()


import os


Batch1 = []
Batch2 = []
Batch3 = []
Batch4 = []

Vars = ["0", "1"]
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

import Create_TC4_PW_Model as NewTC4Model
ThisBatch = Batches[This_Batch]
import subprocess
for Paras in ThisBatch:
    NewTC4Model.create(tuple(Paras))

print("Finished batch " + str(This_Batch))
