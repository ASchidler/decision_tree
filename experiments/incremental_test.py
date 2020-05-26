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
from aaai_encoding import AAAIEncoding

timeout = 1000
memlimit = 2048 * 5

input_path = sys.argv[1]
enc_idx = int(sys.argv[2])
solver_idx = int(sys.argv[3])
strat_idx = int(sys.argv[4])
enable_red = True if sys.argv[5] == "1" else False

tmp_dir = "." if len(sys.argv) == 6 else sys.argv[6]
result_dir = "." if len(sys.argv) == 6 else sys.argv[7]

encodings = [
    DecisionDiagramEncoding,
    DiagramDepthEncoding,
    TreeDepthEncoding,
    TreeEncoding,
    AAAIEncoding
]

solvers = [
    sat_tools.MiniSatSolver,
    sat_tools.GlucoseSolver,
    sat_tools.CadicalSolver
]

strategies = [
    strat.RandomStrategy,
    strat.IncrementalStrategy,
    strat.RetainingStrategy,
    strat.UpdatedRetainingStrategy
]

selected_strategy = strategies[strat_idx]
solver = solvers[solver_idx]
encoding = encodings[enc_idx]

runner = sat_tools.SatRunner(encoding, solver(), base_path=tmp_dir)

# TODO: Start each run with prev size + 1 to increase the chance of getting a SAT result first.
# TODO: Select n examples, s.t. each example increases the number of features required in the key
# TODO: Sanity check that accuracy over test set is indeed 100%

done = set()
test = os.path.split(os.path.normpath(input_path))
out_file = f"results_incremental_{os.path.split(os.path.normpath(input_path))[-1]}_{enc_idx}_{solver_idx}_{enable_red}_{strat_idx}.csv"
out_file = os.path.join(result_dir, out_file)

if not os.path.exists(out_file):
    with open(out_file, "w") as of:
        of.write("Instance;Training Acc.;Test Acc.;Nodes;Depth")
        of.write(os.linesep)

with open(out_file, "r+") as of:
    for i, ln in enumerate(of):
        if i > 0:
            lnc = ln.split(";")
            done.add(lnc[0])

    for fl in os.listdir(input_path):
        if fl.endswith("_training.csv"):
            instance_name = fl[0:-1 * len("_training.csv")]
            if instance_name in done:
                continue

            print("Starting "+instance_name)
            instance = parser.parse(os.path.join(input_path, fl))
            test_instance = parser.parse(os.path.join(input_path, instance_name + "_test.csv"))

            best_tree = None
            best_instance = None
            best_acc = 0

            last_tree = None
            last_instance = None
            improved = False
            strategy = selected_strategy(instance)
            tree_cnt = 0

            start_time = time.time()
            target = encoding.max_instances(instance.num_features, 1)
            while best_acc < 0.99999 and (time.time() - start_time) < timeout:
                # Increment number after all 10 computed trees
                if tree_cnt > 0 and tree_cnt % 10 == 0:
                    target = max(target + 10, int(target * 1.1))

                new_instance = strategy.find_next(best_tree, last_tree, last_instance, target, improved, best_instance)

                new_instance.check_consistency()
                print(f"Using {len(new_instance.examples)} examples")
                if enable_red:
                    bdd_instance.reduce(new_instance)
                    new_instance.check_consistency()

                last_tree,_ = runner.run(new_instance, encoding.new_bound(last_tree, new_instance),
                                       timeout=timeout - (time.time() - start_time), memlimit=memlimit)

                improved = False
                if enable_red:
                    new_instance.unreduce_instance(last_tree)
                last_instance = new_instance

                if last_tree is not None:
                    last_tree.check_consistency()
                    test_acc = last_tree.get_accuracy(new_instance.examples)
                    acc = last_tree.get_accuracy(instance.examples)
                    tree_cnt += 1
                    print(f"Tree found: {acc} Accuracy,  Nodes: {last_tree.get_nodes()}, Depth: {last_tree.get_depth()}, {test_acc} Sanity")

                    if acc > best_acc:
                        best_acc = acc
                        best_tree = last_tree
                        best_instance = new_instance
                        last_accuracy = acc
                        improved = True

            if best_tree is None:
                of.write(f"{instance_name};None")
                print(
                    f"Finished {instance_name}, No tree found")
            else:
                of.write(f"{instance_name};{best_acc};{best_tree.get_accuracy(test_instance.examples)};"
                         f"{best_tree.get_nodes()};{best_tree.get_depth()}{os.linesep}")
                print(
                    f"Finished {instance_name}, Training accuracy: {best_acc}, Test accuracy: {best_tree.get_accuracy(test_instance.examples)}")
            of.flush()

            print("")
            print("")
