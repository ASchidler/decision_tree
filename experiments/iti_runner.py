import os
import sys
import subprocess


class Node:
    def __init__(self):
        self.p = None
        self.ln = None
        self.rn = None


iti_path = "/home/asc/bin/iti"
targets = list(os.listdir(sys.argv[1]))
targets.sort()
sum_test_acc = 0
cnt = 0

for fl in targets:
    if os.path.isdir(os.path.join(sys.argv[1], fl)):
        process = subprocess.Popen([iti_path, fl, "-ltraining", "-qtest", "-f", "-t", "-w"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        output = output.decode('ascii')

        lines = output.split(os.linesep)
        tree = False
        tree_str = None
        leaves = None
        acc = None
        done = True
        max_depth = 0
        node_cnt = 0
        for i, l in enumerate(lines):
            if tree:
                tree = False
                tree_str = l
            if l.startswith("Building tree"):
                tree = True

            if done:
                if len(l.strip()) > 0:
                    node_cnt += 1
                    cDepth = 0
                    for cdc in l:
                        if cdc == " " or cdc == "|":
                            cDepth += 1
                        else:
                            break
                    max_depth = max(max_depth, cDepth)

            if l.startswith("Leaves"):
                stats = l.split()
                leaves = int(stats[1][0:-1])
                acc = float(stats[7][0:-1])
                done = True
        sum_test_acc += acc
        cnt += 1
        print(f"{fl};{node_cnt};{max_depth//3};{acc}")


