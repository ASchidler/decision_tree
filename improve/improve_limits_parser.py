import sys
import os
import re
import collections

pth = sys.argv[1]


def init_coll():
    nd = dict()
    nd["TO"] = []
    nd["NF"] = []
    nd["NT"] = []
    return nd

fls = list(x for x in os.listdir(pth))
results = [init_coll() for _ in range(0, 70)]

for fl in fls:
    if fl.find(".o") < 0:
        continue

    with open(os.path.join(pth, fl)) as inp:
        c_target = None
        start = None
        end = None

        for i, ln in enumerate(inp):
            if i == 0:
                c_target = ln.split(":")[0]
            else:
                ln = ln.replace("\t\t", "\t")
                fields = ln.split("\t")
                if fields[2] in ("NT", "NF", "TO"):
                    d = int(fields[0])
                    lt = int(fields[1])
                    rt = int(fields[5])
                    rd = int(fields[6]) if fields[2] == "NT" else -1
                    tm = float(fields[3])

                    results[d][fields[2]].append((tm, rt,  rd, c_target))

for i in range(0, len(results)):
    r = results[i]
    for k in ["NT", "NF", "TO"]:
        r[k].sort()
        for cvl in r[k]:
            vals = "\t".join((str(x) for x in cvl))
            print(f"{i}\t{k}\t{vals}")
