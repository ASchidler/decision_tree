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

#encoding = DecisionDiagramEncoding
#encoding = TreeEncoding
#encoding = TreeDepthEncoding
encoding = DiagramDepthEncoding

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

strategy = strat.RandomStrategy(instance)
#strategy = strat.IncrementalStrategy(instance)
#strategy = strat.RetainingStrategy(instance)
#strategy = strat.UpdatedRetainingStrategy(instance)
runner = sat_tools.SatRunner(encoding, sat_tools.MiniSatSolver())
improved = False

for _ in range(0, 100):
    new_instance = strategy.find_next(c_tree, last_tree, last_instance, target, improved)

    new_instance.check_consistency()
    print(f"Using {len(new_instance.examples)} examples")
    bdd_instance.reduce(new_instance)
    new_instance.functional_dependencies()
    new_instance.check_consistency()

    last_tree = runner.run(new_instance, encoding.new_bound(last_tree, new_instance))
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

    if acc > 0.99999999:
        break

    new_instance.unreduce_instance(last_tree)
    new_instance.check_consistency()
    last_instance = new_instance
    print(f"Accuracy: {last_accuracy}, This run {acc}")

print("Done")