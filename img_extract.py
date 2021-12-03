import sqlite3
import os


sql = "SELECT Treshold, Mean_F FROM TCF WHERE Name='Alpha' AND Datatype = 'All'"
DB = os.getcwd() + "/Results/Results.db"

Con = sqlite3.connect(DB)
Cur = Con.cursor()

trshf = Cur.execute(sql).fetchall()[:-1]

gw = []
f = []
max = 0
for (trsh, ff) in trshf:
    gw.append(trsh)
    if ff > max:
        max = ff
    f.append(ff)
import numpy as np
import matplotlib.pyplot as plt


maxlist = []
for i in range(len(gw)):
    maxlist.append(max)

fig = plt.figure()
ax = fig.add_subplot()
plt.ylim(0.87, 0.897)

plt.title("Durchschnittlicher F1-Wert bei verschiedenen Schwellenwerten")
ax.set_ylabel("F1-Wert")
ax.set_xlabel("Schwellenwert")
plt.plot(gw, f, "*-", label="F1-Wert")
plt.plot(gw, maxlist, "-", label="Maximum")
plt.legend()
plt.savefig("TCF_A_Tresholds.png")
