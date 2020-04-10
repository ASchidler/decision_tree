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
from BddInstance import BddInstance
import random

instance = parser.parse(sys.argv[1])

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


def compute_tree(c_instance, starting_bound):
    l_bound = 0
    u_bound = sys.maxsize
    c_bound = starting_bound

    enc_file = f"{os.getpid()}.enc"
    model_file = f"{os.getpid()}.model"
    out_file = f"{os.getpid()}.output"

    while l_bound < u_bound:
        with open(enc_file, "w") as f:
            encoding = DecisionDiagramEncoding(f)
            #encoding = TreeEncoding(f)
            #encoding = TreeDepthEncoding(f)
            encoding.encode(c_instance, c_bound)

        with open(out_file, "w") as outf:
            FNULL = open(os.devnull, 'w')
            p1 = subprocess.Popen(['minisat', '-verb=0', enc_file, model_file], stdout=FNULL, stderr=subprocess.STDOUT)

        p1.wait()

        with open(model_file, "r") as f:
            model = parse_minisat(f)
            if model is None:
                l_bound = c_bound + encoding.increment
                c_bound = l_bound + encoding.increment
            else:
                tree = encoding.decode(model, c_instance, c_bound)
                tree.check_consistency()

                u_bound = c_bound
                c_bound -= encoding.increment

        os.remove(enc_file)
        os.remove(model_file)
        os.remove(out_file)

    print(f"Final result: {u_bound}")
    return tree


target = 50
last_instance = None
c_tree = None
last_tree = None
retain = []
last_accuracy = 0
last_index = -1

for _ in range(0, 50):
    new_instance = BddInstance()
    new_instance.num_features = instance.num_features

    if last_instance is not None:
        standins = set()
        for r in retain:
            standins.add(tuple(last_tree.get_path(r.features)))
            new_instance.add_example(r)

        for e in last_instance.examples:
            pth = tuple(last_tree.get_path(e.features))
            if pth not in standins:
                standins.add(pth)
                retain.append(e)
                new_instance.add_example(e)

        print(f"Retained {len(new_instance.examples)} examples")

    i = last_index + 1
    t_target = target // 2
    f_target = target // 2
    while i != last_index:
        if i >= len(instance.examples):
            i = 0
        if f_target == 0 and t_target == 0:
            break

        e = instance.examples[i]

        if last_tree is None or last_tree.decide(e.features) != e.cls:
            if e.cls and t_target > 0:
                new_instance.add_example(e)
                t_target -= 1
            elif not e.cls and f_target > 0:
                new_instance.add_example(e)
                f_target -= 1
        i += 1
    last_index = i
    print(f"Using {len(new_instance.examples)} examples")

    new_instance.reduce_same_features()
    last_tree = compute_tree(new_instance, len(last_tree.nodes) - 1 if last_tree is not None else 7)
    acc = last_tree.get_accuracy(instance.examples)
    if acc >= last_accuracy:
        last_accuracy = acc
        c_tree = last_tree

    new_instance.unreduce_instance(last_tree)
    last_instance = new_instance
    print(f"Accuracy: {last_accuracy}")
