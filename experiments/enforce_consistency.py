import os
import parser
import sys

for r, d, f in os.walk(sys.argv[1]):
    for fl in f:
        if fl.endswith(".csv") and fl != "shuttleM.csv":
            inst = parser.parse_nonbinary(os.path.join(r, fl))
            rm = set()

            for i, e1 in enumerate(inst.examples):
                if i in rm:
                    continue

                for j, e2 in enumerate(inst.examples):
                    if j > i and e1.cls != e2.cls:
                        found = False
                        for f in range(1, inst.num_features + 1):
                            if e1.features[f] != e2.features[f]:
                                found = True
                                break
                        if not found:
                            rm.add(j)

            if len(rm) > 0:
                rm = list(rm)
                rm.sort(reverse=True)
                for crm in rm:
                    inst.examples.pop(crm)

                with open(os.path.join(r, fl), "r") as inst_file:
                    # Keep header
                    hd = inst_file.readline()

                with open(os.path.join(r, fl), "w") as inst_file:
                    inst_file.write(hd)

                    for ce in inst.examples:
                        for f in range(1, inst.num_features+1):
                            inst_file.write(f"{ce.features[f]},")
                        inst_file.write(f"{ce.cls}{os.linesep}")

                print(f"Adapted {fl}")
