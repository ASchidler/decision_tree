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
import sat_tools
from collections import defaultdict

# TODO: Start each run with prev size + 1 to increase the chance of getting a SAT result first.
# TODO: Select n examples, s.t. each example increases the number of features required in the key
# TODO: Sanity check that accuracy over test set is indeed 100%

instance = parser.parse(sys.argv[1])

#encoding = DecisionDiagramEncoding
#encoding = TreeEncoding
encoding = TreeDepthEncoding

target = 50
last_instance = None
c_tree = None
last_tree = None
retain = []
last_accuracy = 0
last_index = -1

#bdd_instance.reduce(instance, optimal=True)
bdd_instance.reduce(instance)
instance.functional_dependencies()
instance.check_consistency()

runner = sat_tools.SatRunner(encoding, sat_tools.MiniSatSolver())
hit_count = {x.id: 0 for x in instance.examples}

for _ in range(0, 50):
    new_instance = BddInstance()
    new_instance.num_features = instance.num_features

    if last_instance is not None:
        path_partition_correct = defaultdict(list)
        path_partition_incorrect = defaultdict(list)

        for e in instance.examples: #last_instance.examples:
            pth = tuple(last_tree.get_path(e.features))
            #result = last_tree.decide(e.features)
            result = c_tree.decide(e.features)

            if result != e.cls:
                path_partition_incorrect[pth].append(e)
            else:
                path_partition_correct[pth].append(e)

        print(f"Found {len(path_partition_correct)} correct paths and {len(path_partition_incorrect)} incorrect paths")
        # Select path representatives
        for k, v in path_partition_correct.items():
            v.sort(key=lambda x: hit_count[x.id])
            c_experiment = v.pop()
            v.clear()
            v.append(c_experiment)
            new_instance.add_example(c_experiment.copy())

        for k, v in path_partition_incorrect.items():
            if k in path_partition_correct:
                repr = path_partition_correct[k][0]
                path_partition_incorrect[k] = [(repr.dist(ce, instance.num_features), ce) for ce in v]
                path_partition_incorrect[k].sort(key=lambda x: (x[0], -1 * hit_count[x[1].id]))
            else:
                path_partition_incorrect[k] = [(-1 * hit_count[ce.id], ce) for ce in v]
                path_partition_incorrect[k].sort(key=lambda x: x[0])

        # Select negative representative
        while len(new_instance.examples) < target:
            for k, v in path_partition_incorrect.items():
                if v:
                    _, c_experiment = v.pop()
                    new_instance.add_example(c_experiment.copy())
                    hit_count[c_experiment.id] += 1

                    if len(new_instance.examples) >= target:
                        break

        print(f"Retained {len(new_instance.examples)} examples")
    else:
        # Initial set

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
                    new_instance.add_example(e.copy())
                    t_target -= 1
                elif not e.cls and f_target > 0:
                    new_instance.add_example(e.copy())
                    f_target -= 1
            i += 1
        last_index = i

    new_instance.check_consistency()
    print(f"Using {len(new_instance.examples)} examples")
    bdd_instance.reduce(new_instance)
    new_instance.functional_dependencies()
    new_instance.check_consistency()

    last_tree = runner.run(new_instance, encoding.new_bound(last_tree, new_instance))
    last_tree.check_consistency()
    print(f"Nodes: {last_tree.get_nodes()}, Depth: {last_tree.get_depth()}")
    acc = last_tree.get_accuracy(instance.examples)
    if acc >= last_accuracy:
        last_accuracy = acc
        c_tree = last_tree

    new_instance.unreduce_instance(last_tree)
    new_instance.check_consistency()
    last_instance = new_instance
    print(f"Accuracy: {last_accuracy}, This run {acc}")
