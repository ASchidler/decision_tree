import argparse as argp
import os
import resource
import sys
import time
from collections import defaultdict
from threading import Timer

from pysat.solvers import Glucose3

import nonbinary.depth_avellaneda_base as base
import nonbinary.depth_avellaneda_sat as nbs
import nonbinary.depth_avellaneda_sat2 as nbs2
import nonbinary.depth_avellaneda_sat3 as nbs3
import nonbinary.depth_avellaneda_smt as nbt
import nonbinary.improve_strategy as improve_strategy
import nonbinary_instance
import tree_parsers
from nonbinary.decision_tree import DecisionTreeNode, DecisionTreeLeaf
from nonbinary.nonbinary_instance import ClassificationInstance

instance_path = "nonbinary/instances"
instance_validation_path = "datasets/validate"

# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (23 * 1024 * 1024 * 1024 // 2, 12 * 1024 * 1024 * 1024))

ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-r", dest="reduce", action="store_true", default=False,
                help="Use support set based reduction. Decreases instance size, but tree may be sub-optimal.")
ap.add_argument("-b", dest="benchmark", action="store_true", default=False,
                help="Benchmark all encodings together.")
ap.add_argument("-c", dest="categorical", action="store_true", default=False,
                help="Treat all features as categorical, this means a one hot encoding as in previous encodings.")
ap.add_argument("-t", dest="time_limit", action="store", default=0, type=int,
                help="The timelimit in seconds.")
ap.add_argument("-z", dest="size", action="store_true", default=False,
                help="Decrease the size as well as the depth.")
ap.add_argument("-e", dest="slim_opt", action="store_true", default=False,
                help="Optimize away extension leaves.")
ap.add_argument("-f", dest="size_first", action="store_true", default=False,
                help="Optimize size before depth.")
ap.add_argument("-d", dest="validation", action="store_true", default=False,
                help="Use data with validation set.")
ap.add_argument("-s", dest="use_smt", action="store_true", default=False)
ap.add_argument("-l", dest="slice", action="store", default=1, type=int,
                help="Which slice to use from the five cross validation sets.")
ap.add_argument("-a", dest="alt_sat", action="store_true", default=False,
                help="Use alternative SAT encoding.")
ap.add_argument("-y", dest="hybrid", action="store_true", default=False,
                help="Use hybrid mode, allowing for <= and =.")
ap.add_argument("-w", dest="weka", action="store_false", default=True,
                help="Use CART instead of WEKA trees.")
ap.add_argument("-m", dest="slim", action="store_true", default=False,
                help="Use local improvement instead of exact results.")
ap.add_argument("-u", dest="maintain", action="store_true", default=False,
                help="Force maintaining of sizes for SLIM.")
ap.add_argument("-o", dest="reduce_categoric", action="store_true", default=False,
                help="In SLIM use full categoric features instead of just single thresholds.")
ap.add_argument("-n", dest="reduce_numeric", action="store_true", default=False,
                help="In SLIM use full numeric features instead of just single thresholds.")
ap.add_argument("-i", dest="limit_idx", action="store", default=1, type=int,
                help="Set of limits.")
ap.add_argument("-x", dest="use_dt", action="store_true", default=False,
                help="Use a decision tree to decide if an instance should be solved.")
ap.add_argument("-g", dest="use_encoding_dt", action="store_true", default=False,
                help="Use a decision tree to decide which encoding to use.")
ap.add_argument("-j", dest="incremental", action="store_true", default=False)
ap.add_argument("-v", dest="recursive", action="store_true", default=False)

args = ap.parse_args()

fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))

fls.sort()

# if args.choices:
#     for i, cf in enumerate(fls):
#         print(f"{i+1}: {cf}")
#     exit(0)
# names = set()
# for i, cf in enumerate(fls):
#     print(f"{i+1} {cf}")
#     names.add(cf.split(".")[0])
# for i, cf in enumerate(names):
#     print(f"{i} {cf}")
try:
    target_instance_idx = int(args.instance)

    if target_instance_idx > len(fls):
        print(f"Only {len(fls)} files are known.")
        exit(1)

    parts = fls[target_instance_idx-1][:-5].split(".")
    target_instance = ".".join(parts[:-1])
    args.slice = int(parts[-1])

except ValueError:
    target_instance = args.instance[:-5] if args.instance.endswith(".names") or args.instance.endswith(".data") else args.instance

print(f"Instance: {target_instance}, {args}")
sys.stdout.flush()

start_time = time.time()

def exit_timeout():
    # TODO: There is a race condition in case the tree is currently changing. This should rarely be the case, as
    # this takes a lot shorter than reduction +
    print(f"Timeout: {time.time() - start_time}")
    #tree.clean(instance, min_samples=args.min_samples)
    print(f"END Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
          f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
          f"Avg. Length: {tree.get_avg_length(instance.examples)}, "
          f"Time: {time.time() - start_time}")

    print(tree.as_string())
    sys.stdout.flush()
    exit(1)

instance, test_instance, validation_instance = nonbinary_instance.parse(instance_path, target_instance, args.slice, use_validation=args.validation)

timer = None
if args.time_limit > 0 and not args.incremental and not args.recursive:
    timer = Timer(args.time_limit * 1.1 - (time.time() - start_time), exit_timeout)
    timer.start()

if args.reduce:
    print(f"{instance.num_features}, {len(instance.examples)} {sum(len(instance.domains[x]) for x in range(1, instance.num_features+1))}")
    instance.reduce_with_key()
    print(f"{instance.num_features}, {len(instance.examples)} {sum(len(instance.domains[x]) for x in range(1, instance.num_features+1))}")

if args.categorical:
    instance.is_categorical = {x for x in range(1, instance.num_features+1)}

tree = None
if args.use_smt:
    enc = nbt
else:
    if args.hybrid:
        enc = nbs3
    elif args.alt_sat:
        enc = nbs2
    else:
        enc = nbs

if args.incremental:
    from nonbinary.incremental.strategy import SupportSetStrategy, SupportSetStrategy2

    tree = base.run_incremental(enc, Glucose3, SupportSetStrategy2(instance), timeout=args.time_limit)
    tree.root.reclassify(instance.examples)
elif args.recursive:
    from nonbinary.incremental.strategy import SupportSetStrategy, SupportSetStrategy2
    leaf_sets = [(list(instance.examples), None)]
    tree = None

    while leaf_sets:
        new_leaf_sets = []
        for c_set, c_root in leaf_sets:
            new_instance = ClassificationInstance()
            new_instance.is_categorical.update(instance.is_categorical)
            for c_e in c_set:
                new_instance.add_example(c_e.copy(new_instance))
            new_instance.finish()
            if len(instance.classes) == 1:
                continue

            new_partial_tree = base.run_incremental(enc, Glucose3, SupportSetStrategy2(new_instance), timeout=args.time_limit, opt_size=args.size)
            if tree is None:
                tree = new_partial_tree
                new_root = tree.root
            else:
                new_root = new_partial_tree.root
                if not new_root.is_leaf:
                    n_n = DecisionTreeNode(new_root.feature, new_root.threshold, c_root, tree,
                                           is_categorical=new_root.is_categorical)
                    n_n.parent = tree.nodes[c_root].parent
                    if tree.nodes[c_root].parent.left.id == c_root:
                        tree.nodes[c_root].parent.left = n_n
                    else:
                        tree.nodes[c_root].parent.right = n_n
                    tree.nodes[c_root] = n_n

                    c_n = [(new_root.left, c_root, True), (new_root.right, c_root, False)]
                    while c_n:
                        c_node, c_parent, c_left = c_n.pop()
                        if c_node.is_leaf:
                            tree.add_leaf(c_node.cls, c_parent, c_left)
                        else:
                            new_n = tree.add_node(c_node.feature, c_node.threshold, c_parent, c_left, c_node.is_categorical)
                            c_n.extend([(c_node.left, new_n.id, True), (c_node.right, new_n.id, False)])
                new_root = tree.nodes[c_root]
                new_root.reclassify(instance.examples)

            if not new_partial_tree.root.is_leaf:
                new_leaves = defaultdict(list)
                for c_e in c_set:
                    _, lf = new_root.decide(c_e)
                    new_leaves[lf.id].append(c_e)
                new_leaf_sets.extend([(x, i) for i, x in new_leaves.items()])
                print(f"Time {time.time() - start_time:.4f}\t"
                      f"Training {tree.get_accuracy(instance.examples):.4f}\t"
                      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
                      f"Depth {tree.get_depth():03}\t"
                      f"Nodes {tree.get_nodes()}\t"
                      f"Avg. Length {tree.get_avg_length(instance.examples)}\t")
        leaf_sets = new_leaf_sets
elif args.slim:
    algo = "w" if args.weka else "c"
    dirs = "validation" if args.validation else "unpruned"
    # if args.categorical:
    #     dirs = "categorical"
    tree = tree_parsers.parse_internal_tree(f"nonbinary/results/trees/{dirs}/{target_instance}.{args.slice}.{algo}.dt")

    parameters = improve_strategy.SlimParameters(tree, instance, enc, Glucose3, args.size, args.slim_opt,
                                                 args.maintain, args.reduce_numeric, args.reduce_categoric,
                                                 args.time_limit, args.use_dt, args.benchmark, args.size_first,
                                                 args.use_encoding_dt)
    if args.use_dt:
        parameters.example_decision_tree = tree_parsers.parse_internal_tree("nonbinary/benchmark_tree.dt")
    elif args.use_encoding_dt:
        parameters.enc_decision_tree = tree_parsers.parse_internal_tree("nonbinary/benchmark_tree_encodings.dt")

    print(f"START Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
          f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
          f"Avg. Length: {tree.get_avg_length(instance.examples)}, "
          f"Time: {time.time() - start_time}")

    improve_strategy.run(parameters, test_instance, limit_idx=args.limit_idx)
else:
    if not args.use_smt:
        tree = base.run(enc, instance, Glucose3, slim=False, opt_size=args.size)
    else:
        tree = nbt.run(instance, opt_size=args.size)

if tree is None:
    print("No tree found.")
    exit(1)

instance.unreduce(tree)
print(f"{instance.num_features}, {len(instance.examples)}")
print(f"END Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
      f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
      f"Avg. Length: {tree.get_avg_length(instance.examples)}, "
      f"Time: {time.time() - start_time}")

print(tree.as_string())


if timer is not None:
    timer.cancel()
