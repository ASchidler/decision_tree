import argparse
import io
import subprocess
import sys
import os
import time
import parser
from tree_node_encoding import TreeEncoding
from diagram_encoding import DecisionDiagramEncoding
from tree_depth_encoding import TreeDepthEncoding
from diagram_depth import DiagramDepthEncoding
from aaai_encoding import AAAIEncoding

instance = parser.parse(sys.argv[1])
test_instance = instance
if sys.argv[1].endswith("_training.csv"):
    test_instance = parser.parse(sys.argv[1][0:-1 * len("_training.csv")] + "_test.csv")

l_bound = 0
u_bound = sys.maxsize
c_bound = 1
stop = False

enc_file = f"{os.getpid()}.enc"
model_file = f"{os.getpid()}.model"
out_file = f"{os.getpid()}.output"


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

tree = None
while l_bound < u_bound:
    with open(enc_file, "w") as f:
        #encoding = DecisionDiagramEncoding(f)
        #encoding = TreeEncoding(f)
        #encoding = TreeDepthEncoding(f)
        #encoding = DiagramDepthEncoding(f)
        encoding = AAAIEncoding(f)
        encoding.encode(instance, c_bound)

    print(f"Num clauses: {encoding.clauses}")

    with open(out_file, "w") as outf:
        p1 = subprocess.Popen(['minisat', '-verb=0', enc_file, model_file])

    p1.wait()

    stop = True
    with open(model_file, "r") as f:
        model = parse_minisat(f)
        if model is None:
            l_bound = c_bound + encoding.increment
            c_bound = l_bound + encoding.increment
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
            c_bound -= encoding.increment

print(f"Final result: {u_bound}")
if tree is not None:
    print(f"Accuracy: {tree.get_accuracy(test_instance.examples)}")

os.remove(enc_file)
os.remove(model_file)
os.remove(out_file)
