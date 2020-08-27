import sys
import os

pth = sys.argv[1]

fls = list(x for x in os.listdir(pth))
results = []
no_result = []

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
                if ln.startswith("Time: Start"):
                    start = (int(fields[3].split()[1]), float(fields[4].split()[1]), float(fields[2].split()[1]))
                else:
                    end = (int(fields[3].split()[1]), float(fields[4].split()[1]), float(fields[2].split()[1]))

        if start is not None:
            if end is not None:
                results.append((c_target, start[0], start[1], start[2], end[0], end[1], end[2]))
            else:
                no_result.append((c_target, start[0], start[1], start[2]))

results.sort()
for n in no_result:
    print(n)
for r in results:
    print(r)


