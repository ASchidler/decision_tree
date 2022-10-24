import os
from collections import defaultdict
from shutil import move

pth = "nonbinary/instances/"

files = list(os.listdir(pth))
file_name = defaultdict(list)

for c_f in files:
    if c_f.endswith(".data"):
        base_name = ".".join(c_f.split(".")[:-2])
        file_name[base_name].append(c_f)

for c_base, c_files in file_name.items():
    print(f"Processing {c_base}")
    lines = defaultdict(lambda: defaultdict(int))
    for c_f in c_files:
        for _, c_line in enumerate(open(pth+ c_f)):
            fields = c_line.split(",")
            head = ",".join(fields[:-1])
            cls = fields[-1]
            lines[head][cls] += 1

    conflicts = dict()
    for c_header, c_v in lines.items():
        if len(c_v) > 1:
            _, new_cls = max((v, k) for k, v in c_v. items())
            conflicts[c_header] = new_cls

    if len(conflicts) > 0:
        print("Found conflict")

        for c_f in c_files:
            found = False
            with open(pth + c_f + ".tmp", "w") as nf:
                for _, c_line in enumerate(open(pth + c_f)):
                    fields = c_line.split(",")
                    head = ",".join(fields[:-1])
                    cls = fields[-1]

                    if head in conflicts and cls != conflicts[head]:
                        found = True
                        nf.write(head)
                        nf.write(",")
                        nf.write(conflicts[head])
                    else:
                        nf.write(c_line)

                if found:
                    print(f"Fixed {c_f}")
                    os.remove(pth + c_f)
                    move(pth + c_f + ".tmp", pth + c_f)
                else:
                    os.remove(pth + c_f + ".tmp")


