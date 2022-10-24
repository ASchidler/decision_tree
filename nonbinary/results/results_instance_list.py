import os
import sys

import nonbinary.nonbinary_instance as inst

ignores = set()
with open("ignore.txt") as ig:
    for cig in ig:
        ignores.add(cig.strip())

instances = {}

for c_file in os.listdir("../instances"):
    if c_file.endswith(".data"):
        if int(c_file[-6:-5]) == 1:
            instance1, _, _ = inst.parse("../instances", c_file[:-7], int(c_file[-6:-5]), use_validation=False, use_test=False)
            instance2, _, _ = inst.parse("../instances", c_file[:-7], int(c_file[-6:-5]), use_validation=False,
                                         use_test=True)
            instance3, _, _ = inst.parse("../instances", c_file[:-7], int(c_file[-6:-5]), use_validation=True,
                                         use_test=True)
            instances[c_file[:-7]] = (len(instance1.examples), len(instance2.examples), len(instance3.examples),
                                      instance1.num_features, len(instance1.classes), c_file[:-7] in ignores)

print("\\begin{table}")
print("\\begin{tabular}{l rrr rr l")
#print("Instance & $\Card{E}$ & $\Card{E}$ (with Test) & $\Card{E}$ (with Test and Validation) & $\Card{F}$ & $\Card{C}$ & Solved\\\\")
print("Instance & $\Card{E}$ $\Card{F}$ & $\Card{C}$ & Solved & Instance & $\Card{E}$ $\Card{F}$ & $\Card{C}$ & Solved\\\\")
keys = sorted(instances.keys(), key=lambda x: x.upper())
endidx = len(keys) // 2 + (0 if len(keys) % 2 == 0 else 1)
for cidx in range(0, endidx): # sorted(instances.keys(), key=lambda x: x.upper()):
    for ci, ck in enumerate([keys[cidx], keys[cidx+endidx] if cidx + endidx < len(keys) else None]):
        if ci == 1:
            sys.stdout.write("&")
        if ck is not None:
            cl = instances[ck]
            sys.stdout.write(ck.replace("_", " "))
            for entry in [cl[0], cl[3], cl[4]]:
                sys.stdout.write(f"&{entry}")
            sys.stdout.write(f"&{'' if not cl[5] else 'x'}")
        else:
            sys.stdout.write("&&&&")
    print("\\\\")
print("\\end{tabular")
print("\\end{table}")
