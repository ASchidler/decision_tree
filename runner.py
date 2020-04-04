import argparse
import io
import subprocess
import sys
import os
import time
import parser
from tree_encoding import TreeEncoding

instance = parser.parse(sys.argv[1])
l_bound = 0
u_bound = sys.maxsize
c_bound = 15
stop = False


def parse_minisat(f):
    first = f.readline()
    if first.startswith("UNSAT"):
        return None

    # TODO: This could be faster using a list...
    model = {}
    vars = f.readline().split()
    for v in vars:
        val = int(v)
        model[abs(val)] = val > 0

    return model


while not stop:
    with open("tmp.enc", "w") as f:
        encoding = TreeEncoding(f)
        encoding.encode(instance, c_bound)

    with open("tmp.txt", "w") as outf:
        p1 = subprocess.Popen(['minisat', '-verb=0', "tmp.enc", "tmpout.txt"])

    p1.wait()

    with open("tmpout.txt", "r") as f:
        model = parse_minisat(f)
        tree = encoding.decode(model, instance, c_bound)
        if tree is not None:
            stop = True
            tree.check_tree_consistency()

            # Verify tree
            total = 0
            correct = 0
            for e in instance.examples:
                decision = tree.decide(e.features)
                total += 1
                if decision != e.cls:
                    print(f"ERROR: Decision mismatch, should be {e.cls} but is {decision}.")
                else:
                    correct += 1
            print(f"Accuracy: {correct/total}")

