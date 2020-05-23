import argparse
import io
import subprocess
import sys
import os
import time
import parser
from tree_node_encoding import TreeEncoding
from diagram_encoding import DecisionDiagramEncoding
from diagram_depth import DiagramDepthEncoding
from tree_depth_encoding import TreeDepthEncoding
from bdd_instance import BddInstance
import bdd_instance
import random
import sat_tools
from collections import defaultdict
import strategies.strategies as strat

# TODO: Start each run with prev size + 1 to increase the chance of getting a SAT result first.
# TODO: Select n examples, s.t. each example increases the number of features required in the key
# TODO: Sanity check that accuracy over test set is indeed 100%

instance = parser.parse(sys.argv[1])
test_instance = instance
if sys.argv[1].endswith("_training.csv"):
    test_instance = parser.parse(sys.argv[1][0:-1 * len("_training.csv")] + "_test.csv")

#encoding = DecisionDiagramEncoding
#encoding = TreeEncoding
encoding = TreeDepthEncoding
#encoding = DiagramDepthEncoding

target = 50
last_instance = None
c_tree = None
last_tree = None
retain = []
last_accuracy = 0
last_index = -1
limit = 20

#bdd_instance.reduce(instance, optimal=True)
bdd_instance.reduce(instance)
instance.functional_dependencies()
instance.check_consistency()

strategy = strat.RandomStrategy(instance)
#strategy = strat.IncrementalStrategy(instance)
#strategy = strat.RetainingStrategy(instance)
#strategy = strat.UpdatedRetainingStrategy(instance)
runner = sat_tools.SatRunner(encoding, sat_tools.CadicalSolver()) #sat_tools.GlucoseSolver()) #sat_tools.MiniSatSolver())
improved = False

start_time = time.time()
while (time.time() - start_time) < limit:
    new_instance = strategy.find_next(c_tree, last_tree, last_instance, target, improved)

    new_instance.check_consistency()
    print(f"Using {len(new_instance.examples)} examples")
    bdd_instance.reduce(new_instance)
    new_instance.functional_dependencies()
    new_instance.check_consistency()

    last_tree = runner.run(new_instance, encoding.new_bound(last_tree, new_instance), timeout=limit - (time.time() - start_time))

    if last_tree is not None:
        last_tree.check_consistency()
        print(f"Nodes: {last_tree.get_nodes()}, Depth: {last_tree.get_depth()}")
        test_acc = last_tree.get_accuracy(new_instance.examples)
        print(test_acc)
        acc = last_tree.get_accuracy(instance.examples)
        improved = False
        if acc > last_accuracy:
            last_accuracy = acc
            c_tree = last_tree
            improved = True

        print(f"Accuracy: {last_accuracy}, This run {acc}")
        if acc > 0.99999999:
            break

    new_instance.unreduce_instance(last_tree)
    new_instance.check_consistency()
    last_instance = new_instance

print("Done")
if c_tree is not None:
    print(f"Accuracy: {c_tree.get_accuracy(test_instance.examples)}")
