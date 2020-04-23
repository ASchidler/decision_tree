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
from bdd_instance import BddInstance
import bdd_instance
import random

# TODO: Start each run with prev size + 1 to increase the chance of getting a SAT result first.
# TODO: Select n examples, s.t. each example increases the number of features required in the key

instance = parser.parse(sys.argv[1])

#encoding = DecisionDiagramEncoding
encoding = TreeEncoding
#encoding = TreeDepthEncoding

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
    l_bound = encoding.lb()
    u_bound = sys.maxsize
    c_bound = starting_bound

    enc_file = f"{os.getpid()}.enc"
    model_file = f"{os.getpid()}.model"
    out_file = f"{os.getpid()}.output"

    while l_bound < u_bound:
        print(f"Running with limit {c_bound}")
        with open(enc_file, "w") as f:
            inst_encoding = encoding(f)
            inst_encoding.encode(c_instance, c_bound)

        with open(out_file, "w") as outf:
            FNULL = open(os.devnull, 'w')
            p1 = subprocess.Popen(['minisat', '-verb=0', enc_file, model_file], stdout=FNULL, stderr=subprocess.STDOUT)

        p1.wait()

        with open(model_file, "r") as f:
            model = parse_minisat(f)
            if model is None:
                l_bound = c_bound + inst_encoding.increment
                c_bound = l_bound
            else:
                tree = inst_encoding.decode(model, c_instance, c_bound)
                tree.check_consistency()

                u_bound = c_bound
                c_bound -= inst_encoding.increment

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

bdd_instance.reduce(instance)
instance.functional_dependencies()
instance.check_consistency()

for _ in range(0, 50):
    new_instance = BddInstance()
    new_instance.num_features = instance.num_features

    if last_instance is not None:
        standins = set()
        for r in retain:
            standins.add(tuple(last_tree.get_path(r.features)))
            new_instance.add_example(bdd_instance.BddExamples(r.features, r.cls))

        for e in instance.examples: #last_instance.examples:
            pth = tuple(last_tree.get_path(e.features))
            if pth not in standins:
                standins.add(pth)
                retain.append(e)
                new_instance.add_example(bdd_instance.BddExamples(e.features, e.cls))

        print(f"Retained {len(new_instance.examples)} examples")
    new_instance.check_consistency()
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
                new_instance.add_example(bdd_instance.BddExamples(e.features, e.cls))
                t_target -= 1
            elif not e.cls and f_target > 0:
                new_instance.add_example(bdd_instance.BddExamples(e.features, e.cls))
                f_target -= 1
        i += 1
    last_index = i
    new_instance.check_consistency()
    print(f"Using {len(new_instance.examples)} examples")
    bdd_instance.reduce(new_instance)
    new_instance.functional_dependencies()
    new_instance.check_consistency()

    last_tree = compute_tree(new_instance, encoding.new_bound(last_tree, new_instance))
    acc = last_tree.get_accuracy(instance.examples)
    if acc >= last_accuracy:
        last_accuracy = acc
        c_tree = last_tree

    new_instance.unreduce_instance(last_tree)
    new_instance.check_consistency()
    last_instance = new_instance
    print(f"Accuracy: {last_accuracy}, This run {acc}")
