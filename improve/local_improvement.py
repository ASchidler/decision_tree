import sys
import parser
import os
import decision_tree
import improve.improve_depth_first as df
import improve.improve_leaf_first as lf
import improve.improve_random as rf
import improve.improve_size as sf
import random
from improve.tree_parsers import parse_weka_tree, parse_iti_tree
import resource
import argparse as argp

random.seed = 1
# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, 17 * 1024 * 1024 * 1024))

tree_path = "datasets/trees"
instance_path = "datasets/split"

ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-a", dest="alg", action="store", default=0, choices=[0, 1, 2, 3], type=int,
                help="Which decision tree algorithm to use (0=C4.5, 1=ITI, 2=CART, 3=SAT).")
ap.add_argument("-l", dest="limit_idx", action="store", default=1, type=int,
                help="The index for the set of limits used for selecting sub-trees.")
ap.add_argument("-e", dest="print_tree", action="store_true", default=False,
                help="Export decision trees in dot format.")
ap.add_argument("-p", dest="load_pruned", action="store", default=0,  type=int,
                help="Choose pruned decision tree with this index, 0 means unpruned tree.")
ap.add_argument("-m", dest="method_prune", action="store", default=0, choices=[0, 1, 2, 3], type=int,
                help="Pruning method to use (0=None, 1=C4.5, 2=Cost Complexity, 3=Reduced Error).")
ap.add_argument("-i", dest="immediate_prune", action="store_true", default=False,
                help="Prune each sub-tree after computation.")
ap.add_argument("-t", dest="time_limit", action="store", default=0, type=int,
                help="The timelimit in seconds. Note that an elapsed timelimit will not cancel the current SAT call."
                     "Depending on the used limits there is an imprecision of several minutes.")

args = ap.parse_args()

try:
    target_instance_idx = int(args.instance)
    fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
    fls.sort()

    if target_instance_idx > len(fls):
        print(f"Only {len(fls)} files are known.")
        exit(1)

    target_instance = fls[target_instance_idx-1][:-5]
except ValueError:
    target_instance = args.instance[:-5] if args.instance.endswith(".names") or args.instance.endswith(".data") else args.instance

tree_infix = ""
if args.load_pruned > 0:
    tree_infix = f".p{args.load_pruned}"

training_instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)
test_instance = parser.parse(os.path.join(instance_path, target_instance + ".test"), has_header=False)

if args.alg == 0:
    tree = parse_weka_tree(os.path.join(tree_path, target_instance + tree_infix + ".tree"), training_instance)
elif args.alg == 1:
    tree = parse_iti_tree(os.path.join(tree_path, target_instance + tree_infix + ".iti"), training_instance)
elif args.alg == 2:
    tree = parse_iti_tree(os.path.join(tree_path, target_instance + tree_infix + ".cart"), training_instance)
else:
    raise RuntimeError(f"Unknown DT algorithm {args.alg}")

# Parse tree

if args.print_tree:
    with open("input_tree2.gv", "w") as f:
        f.write(decision_tree.dot_export(tree))

print(f"{target_instance}: Features {training_instance.num_features}\tExamples {len(training_instance.examples)}\t"
      f"Optimize 'Depth'\tHeuristic {'Weka' if args.alg == 0 else ('ITI' if args.alg == 1 else 'CART')}")

print(f"Time: Start\t\t"
      f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
      f"Depth {tree.get_depth():03}\t"
      f"Avg {tree.get_avg_depth():03.4f}\t"
      f"Nodes {tree.get_nodes()}")

df.run(tree, training_instance, test_instance, limit_idx=args.limit_idx, pt=args.print_tree, timelimit=args.time_limit)

print(f"Time: End\t\t"
      f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
      f"Depth {tree.get_depth():03}\t"
      f"Avg {tree.get_avg_depth():03.4f}\t"
      f"Nodes {tree.get_nodes()}")
