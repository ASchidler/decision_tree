import sys
import os
import re

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

c_file = None
c_results = None
new_results = []

for r in results:
    if c_file is not None:
        if r[0].startswith(c_file):
            c_results.append(r)
            continue
        else:
            n_r = [c_file]
            for i in range(1, len(r)):
                val = 0
                cnt = 0
                for c_l in c_results:
                    val += c_l[i]
                    cnt += 1
                n_r.append(val / cnt)

            new_results.append([*n_r, len(c_results)])
            c_file = None
            c_results = None

    if re.search("[0-4]$", r[0]):
        c_file = r[0][:-2]
        c_results = [r]
    else:
        new_results.append([*r, 1])

for r in new_results:
    print("\t".join([f"{x}" for x in r]))


