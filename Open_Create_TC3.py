This_Batch = 0


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


ThisBatch = Batches[This_Batch]
Func = os.getcwd() + "/Create_TC3_Model.py"

for Paras in ThisBatch:
    NewTarget = ""
    for letter in Paras:
        NewTarget += " " + letter
    os.system(Func + NewTarget)
print("Finished")
