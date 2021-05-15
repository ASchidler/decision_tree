import sys
import time

import parser
import os
import decision_tree
import improve.improve_depth_first as df
import improve.improve_easy_first as ef
import random
from improve.tree_parsers import parse_weka_tree, parse_iti_tree, parse_internal_tree
import resource
import argparse as argp
from class_instance import split
import pruning
from threading import Timer
import signal

start_time = time.time()
random.seed = 1
# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (12 * 1024 * 1024 * 1024, 25 * 1024 * 1024 * 1024 // 2))

tree_path = "datasets/trees"
instance_path = "datasets/split"
tree_validation_path = "datasets/validate-trees"
instance_validation_path = "datasets/validate"
sat_path = "datasets/rec"
validation_ratios = [0, 20, 30]
heuristics = {0: "Weka", 1: "ITI", 2: "CART", 3: "SAT"}

ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-c", dest="choices", action="store_true", default=False,
                help="List the known instances with their id")
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
ap.add_argument("-r", dest="ratio", action="store", default=0.3, type=float,
                help="Ratio used for pruning. The semantics depends on the pruning method.")
ap.add_argument("-s", dest="min_samples", action="store", default=1, type=int,
                help="The minimum number of samples per leaf.")
ap.add_argument("-d", dest="validation", action="store", default=0, type=int,
                help="Use data with validation set, 1=20% holdout, 2=30% holdout.")
ap.add_argument("-z", dest="size", action="store_true", default=False,
                help="Decrease the size as well as the depth.")
ap.add_argument("-y", dest="easy", action="store_true", default=False,
                help="Use easy operations first")
ap.add_argument("-g", dest="strategy", action="store", default=0, type=int,
                help="Strategy SAT tree to load.")

args = ap.parse_args()

if args.validation == 0:
    fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
else:
    fls = list(
        x for x in os.listdir(instance_validation_path) if x.endswith(f"{validation_ratios[args.validation]}.data"))

fls.sort()

if args.choices:
    for i, cf in enumerate(fls):
        print(f"{i+1}: {cf}")
    exit(0)

try:
    target_instance_idx = int(args.instance)

    if target_instance_idx > len(fls):
        print(f"Only {len(fls)} files are known.")
        sys.stdout.flush()
        exit(1)

    target_instance = fls[target_instance_idx-1][:-5]
except ValueError:
    target_instance = args.instance[:-5] if args.instance.endswith(".names") or args.instance.endswith(".data") else args.instance

tree_infix = ""
if args.load_pruned > 0:
    tree_infix = f".p{args.load_pruned}"

if args.validation > 0:
    instance_path = instance_validation_path
    tree_path = tree_validation_path

training_instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)
test_instance = parser.parse(os.path.join(instance_path, target_instance + ".test"), has_header=False)

if args.alg == 0:
    tree = parse_weka_tree(os.path.join(tree_path, target_instance + tree_infix + ".tree"), training_instance)
elif args.alg == 1:
    tree = parse_iti_tree(os.path.join(tree_path, target_instance + tree_infix + ".iti"), training_instance)
elif args.alg == 2:
    tree = parse_internal_tree(os.path.join(tree_path, target_instance + tree_infix + ".cart"), training_instance)
elif args.alg == 3:
    tree = parse_internal_tree(os.path.join(sat_path, target_instance + f".20{args.strategy}.dt"), training_instance)
else:
    raise RuntimeError(f"Unknown DT algorithm {args.alg}")

# Parse tree

if args.print_tree:
    with open("input_tree2.gv", "w") as f:
        f.write(decision_tree.dot_export(tree))

print(f"{target_instance}: Features {training_instance.num_features}\tExamples {len(training_instance.examples)}\t"
      f"Optimize 'Depth'\tHeuristic {heuristics[args.alg]}")


def exit_timeout():
    # TODO: There is a race condition in case the tree is currently changing. This should rarely be the case, as
    # this takes a lot shorter than reduction +
    print(f"Timeout: {time.time() - start_time}")
    tree.clean(training_instance, min_samples=args.min_samples)
    print(f"Time: End\t\t"
          f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
          f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
          f"Depth {tree.get_depth():03}\t"
          f"Avg {tree.get_avg_depth():03.4f}\t"
          f"Nodes {tree.get_nodes()}")
    print(tree.as_string())
    exit(1)


timer = None
if args.time_limit > 0:
    timer = Timer(args.time_limit * 1.1 - (time.time() - start_time), exit_timeout)
    timer.start()

print(f"Time: Start\t\t"
      f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
      f"Depth {tree.get_depth():03}\t"
      f"Avg {tree.get_avg_depth():03.4f}\t"
      f"Nodes {tree.get_nodes()}")


# Although there are a lot of safeguards there is some edge condition that causes a segfault (probably due to Memout)
def sig_handler(signum, frame):
    print("ERROR: Segfault")
    print(f"Time: End\t\t"
          f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
          f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
          f"Depth {tree.get_depth():03}\t"
          f"Avg {tree.get_avg_depth():03.4f}\t"
          f"Nodes {tree.get_nodes()}")

    print(tree.as_string())
    exit(1)


signal.signal(signal.SIGSEGV, sig_handler)

if args.method_prune != 3:
    if args.easy:
        ef.run(tree, training_instance, test_instance, limit_idx=args.limit_idx, pt=args.print_tree,
               timelimit=0 if args.time_limit == 0 else args.time_limit - (time.time() - start_time), opt_size=args.size)
    else:
        df.run(tree, training_instance, test_instance, limit_idx=args.limit_idx, pt=args.print_tree,
               timelimit=0 if args.time_limit == 0 else args.time_limit - (time.time() - start_time), opt_size=args.size)
    if args.method_prune == 1:
        pruning.prune_c45(tree, training_instance, args.ratio, m=args.min_samples)
    else:
        tree.clean(training_instance, min_samples=args.min_samples)
else:
    new_training, holdout = split(training_instance, ratio_splitoff=args.ratio)
    if args.easy:
        ef.run(tree, new_training, test_instance, limit_idx=args.limit_idx, pt=args.print_tree,
               timelimit=0 if args.time_limit == 0 else args.time_limit - (time.time() - start_time), opt_size=args.size)
    else:
        df.run(tree, new_training, test_instance, limit_idx=args.limit_idx, pt=args.print_tree,
               timelimit=0 if args.time_limit == 0 else args.time_limit - (time.time() - start_time), opt_size=args.size)

    tree.clean(new_training, min_samples=args.min_samples)
    pruning.prune_reduced_error(tree, holdout)

print(f"Time: End\t\t"
      f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
      f"Depth {tree.get_depth():03}\t"
      f"Avg {tree.get_avg_depth():03.4f}\t"
      f"Nodes {tree.get_nodes()}")

print(tree.as_string())

if timer:
    timer.cancel()
