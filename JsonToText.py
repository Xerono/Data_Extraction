Jsons = ["LandUse450.json", "Management332.json", "Properties1.json"]



import json
import os


Filesfolder = os.getcwd() + "/Files/"
with open(Filesfolder + Jsons[2], encoding = "UTF-8") as file:
    Props = json.load(file)

Textures = []
for item in Props["children"][1]["children"][0]["children"][5]["children"][0]["children"]:
    CurText = item["text"]
    CurText = CurText.split(" (")[0]
    Textures.append(CurText)




for item in Props["children"][1]["children"][0]["children"][5]["children"][1]["children"]:
    CurText = item["text"]
    CurText = CurText.split(" (")[0]
    Textures.append(CurText)
    
#print(Textures)


with open(Filesfolder + Jsons[1], encoding = "UTF-8") as file:
    Manag = json.load(file)


Crops = []
Targets = [6, 7, 8, 9]
for trgt in Targets:
    for item in Manag["children"][8]["children"][trgt]["children"]:
        for crop in item["displayText"].split("/"):
            Crops.append(crop)
        if "children" in item.keys():
            for newitem in item["children"]:
                for newcrop in newitem["displayText"].split("/"):
                    Crops.append(newcrop)
            if "children" in newitem.keys():
                for newnewitem in newitem["children"]:
                    for newnewcrop in newnewitem["displayText"].split("/"):
                        Crops.append(newnewcrop)                    
Crops = set(Crops)

with open(Filesfolder + "Texturelist.txt", "w") as target:
    for txt in Textures:
        target.write(txt + "\n")
with open(Filesfolder + "Cropslist.txt", "w") as target:
    for txt in Crops:
        target.write(txt + "\n")   
