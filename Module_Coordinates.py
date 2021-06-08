
import re
def find_coordinates(Par):
    Sixers = []
    Eighters = []
    Errors = []
    NotFound = []
    someerror = False
    # Find xx°xx'N
    rege1 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,10}[0-9][0-9].{1,4}(?:W|E)"#.{1,1}"
    # Find xx°xx'xx''N
    rege2 = "[0-9][0-9].{1,4}[0-9][0-9].{1,4}[0-9][0-9].{1,4}(?:N|S).{1,5}[0-9].{0,6}[0-9][0-9].{1,4}(?:W|E)"#.{1,1}"
    regelist = [rege1, rege2]
    res = []
    
    for regex in regelist:
        res = res + re.findall(regex, Par)
    if len(res)>0:
        for potcord in res:
            pc = convert(potcord)
            for item in pc:
                if len(item)>=3:
                    someerror = True
            if not someerror:
                if len(pc) == 6 and (pc[2] == "N" or pc[2] == "S") and (pc[5] == "W" or pc[5] == "E"):
                    Sixers.append((pc, potcord, Par))
                else:
                    if len(pc) == 8 and (pc[3] == "N" or pc[3] == "S") and (pc[7] == "W" or pc[7] == "E"):
                        Eighters.append((pc, potcord, Par))
                    else:
                        someerror = True
            if someerror:
                Errors.append((pc, potcord, Par))
    else:
        if len(Par)>0:
            NotFound.append(Par)
    return (Sixers, Eighters, NotFound, Errors)

def convert(potcord):
    potcord = removal(potcord)
    return potcord.split(" ")


def remove_middle_twos(potcord):
    reg = "[0-9][0-9]2[0-9][0-9]"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord


def remove_middle_eights(potcord):
    reg = "[0-9][0-9]8[0-9][0-9]"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_end_eights(potcord):
    reg = "[0-9][0-9]8(?:N|S|W|E)"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_middle_nines(potcord):
    reg = "[0-9][0-9]9[0-9][0-9]"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_end_nines(potcord):
    reg = "[0-9][0-9]9(?:N|S|W|E)"
    results = re.findall(reg, potcord)
    for found in results:
        foundlist = list(found)
        foundlist[2] = " "
        potcord = potcord.replace(found, "".join(foundlist))
    return potcord

def remove_additional_spaces(potcord):
    while "  " in potcord:
        potcord = potcord.replace("  ", " ")
    if potcord[-1] == " ":
        potcord = potcord[:-1]
    if potcord[0]== " ":
        potcord = potcord[1:]
    return potcord

def remove_additional_w(potcord):
    pc = potcord.split(" ")
    potc = ""
    for part in pc:
        partlist = list(part)
        if len(partlist)==4:
            if partlist[2] == "W":
                partlist[2] = " "
        potco = ""
        for item in partlist:
            potco = potco + item
            
        potc = potc + " " + potco
    return potc

def split_triples(potcord):
    pc = potcord.split(" ")
    returnlist = []
    for part in pc:
        found = False
        partlist = list(part)
        if len(partlist) == 3:
            if partlist[2] in ["N", "S", "W", "E"]:
                p1 = partlist[:2]
                p2 = partlist[2:]
                found = True
        if len(partlist) == 4:
            if partlist[3] in ["N", "S", "W", "E"]:
                p1 = partlist[:2]
                p2 = partlist[3:]
                found = True
        if found:
            pa1 = ""
            for char in p1:
                pa1 = pa1 + char
            pa2 = ""
            for char in p2:
                pa2 = pa2 + char
            returnlist.append(pa1)
            returnlist.append(pa2)
        else:
            returnlist.append(part)
    poco = ""
    for item in returnlist:
        poco = poco + " " + item
    return poco

def removal(potcord):
    potcord = remove_middle_twos(potcord)
    potcord = remove_middle_eights(potcord)
    potcord = remove_end_eights(potcord)
    potcord = remove_middle_nines(potcord)
    potcord = remove_end_nines(potcord)
    potcord = re.sub(r'[^0-9NSWE]', " ", potcord)
    potcord = remove_additional_w(potcord)
    potcord = split_triples(potcord)
    potcord = remove_additional_spaces(potcord)
    return potcord


