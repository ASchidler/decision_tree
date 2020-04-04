import argparse
import io
import subprocess
import sys
import os
import time
import parser
from tree_encoding import TreeEncoding
from diagram_encoding import DecisionDiagramEncoding

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


while l_bound < u_bound:
    with open("tmp.enc", "w") as f:
        encoding = DecisionDiagramEncoding(f) # TreeEncoding(f)
        encoding.encode(instance, c_bound)
    print(f"Num clauses: {encoding.clauses}")

    with open("tmp.txt", "w") as outf:
        p1 = subprocess.Popen(['minisat', '-verb=0', "tmp.enc", "tmpout.txt"])

    p1.wait()

    stop = True
    with open("tmpout.txt", "r") as f:
        model = parse_minisat(f)
        if model is None:
            l_bound = c_bound + 1
            c_bound = l_bound + 1
        else:
            tree = encoding.decode(model, instance, c_bound)
            tree.check_consistency()

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
            u_bound = c_bound
            c_bound -= 1

print(f"Final result: {u_bound}")
