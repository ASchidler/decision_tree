import subprocess
import sys
import os
import time

import decision_tree
import class_instance
import parser
import sat.size_narodytska as sn
import sat.depth_avellaneda as da
import sat.depth_partition as dp
import sat.switching_encoding as se
from pysat.solvers import Glucose3
import argparse as argp
import incremental.entropy_strategy as es
import incremental.random_strategy as rs
import incremental.maintain_strategy as ms
import limits
import resource
import sat.encoding_base as eb
import incremental.heuristic as heur

instance_path = "datasets/split"
instance_validation_path = "datasets/validate"
validation_ratios = [0, 20, 30]

# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (23 * 1024 * 1024 * 1024 // 2, 12 * 1024 * 1024 * 1024))

encodings = [se, da, dp, sn]
strategies = [es.EntropyStrategy2, rs.RandomStrategy, ms.MaintainingStrategy]


ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-e", dest="encoding", action="store", default=0, choices=[0, 1, 2, 3], type=int,
                help="The encoding to use (0=switching, 1=Avellaneda, 2=SchidlerSzeider, 3=NarodytskaEtAl")
ap.add_argument("-m", dest="mode", action="store", default=0, choices=[0, 1, 2], type=int,
                help="The induction mode (0=All at once, 1=incremental, 2=recursive)."
                )
ap.add_argument("-r", dest="reduce", action="store_true", default=False,
                help="Use support set based reduction. Decreases instance size, but tree may be sub-optimal.")
ap.add_argument("-t", dest="time_limit", action="store", default=900, type=int,
                help="The timelimit in seconds.")
ap.add_argument("-s", dest="strategy", action="store", default=0, choices=[0, 1, 2], type=int,
                help="The strategy to use for incremental, recursive mode (0=entropy, 1=random, 2=maintaining)."
                )
ap.add_argument("-z", dest="size", action="store_true", default=False,
                help="Decrease the size as well as the depth.")
ap.add_argument("-d", dest="validation", action="store", default=0, type=int,
                help="Use data with validation set, 1=20% holdout, 2=30% holdout.")

args = ap.parse_args()

if args.validation == 0:
    fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
else:
    fls = list(
        x for x in os.listdir(instance_validation_path) if x.endswith(f"{validation_ratios[args.validation]}.data"))

fls.sort()

# if args.choices:
#     for i, cf in enumerate(fls):
#         print(f"{i+1}: {cf}")
#     exit(0)

try:
    target_instance_idx = int(args.instance)

    if target_instance_idx > len(fls):
        print(f"Only {len(fls)} files are known.")
        exit(1)

    target_instance = fls[target_instance_idx-1][:-5]
except ValueError:
    target_instance = args.instance[:-5] if args.instance.endswith(".names") or args.instance.endswith(".data") else args.instance

print(f"{target_instance}")

start_time = time.time()
if args.validation == 0:
    instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)
else:
    instance = parser.parse(os.path.join(instance_validation_path, target_instance + ".data"), has_header=False)
test_instance = instance
if os.path.exists(args.instance[:-4]+"test"):
    test_instance = parser.parse(args.instance[:-4] + "test")

encoding = encodings[args.encoding]
strat = strategies[args.strategy]

if args.reduce:
    class_instance.reduce(instance)

if args.mode == 0:
    tree = eb.run(encoding, instance, Glucose3, timeout=args.time_limit, opt_size=args.size, check_mem=False)
    # encoding.run(instance, Glucose3, timeout=args.time_limit, opt_size=args.size, check_mem=False)
elif args.mode == 1:
    strategy = strat(instance)
    strategy.extend(5)
    tree = eb.run_incremental(encoding, instance, Glucose3, strategy, args.time_limit, limits.size_limit, opt_size=args.size)
elif args.mode == 2:
    strategy = strat(instance)
    strategy.extend(5)
    tree = eb.run_incremental(encoding, instance, Glucose3, strategy, args.time_limit, limits.size_limit, opt_size=args.size)
    #tree = encoding.run_limited(Glucose3, strategy, limits.size_limit, limits.sample_limit_short, start_bound=len(limits.sample_limit_short)-2, go_up=False)

    def add_nodes(c_root):
        if not c_root.is_leaf:
            for c_child, c_pol in [(c_root.left, True), (c_root.right, False)]:
                tree.nodes.append(None)
                if c_child.is_leaf:
                    tree.add_leaf(len(tree.nodes)-1, c_root.id, c_pol, c_child.cls)
                else:
                    new_id = len(tree.nodes) - 1
                    c_child.id = new_id
                    tree.add_node(new_id, c_root.id, c_child.feature, c_pol)
                    add_nodes(c_child)

    changed = True
    while changed:
        print(f"Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
              f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}")
        # TODO: It is more efficient to just reassign the samples that were associated with the replaced leaf
        assigned = tree.assign_samples(instance)
        changed = False
        c_len = len(tree.nodes)
        for c_node_idx in range(0, c_len):
            c_node = tree.nodes[c_node_idx]
            if c_node and c_node.is_leaf:
                misclassified = False
                for c_e in assigned[c_node.id]:
                    if instance.examples[c_e].cls != c_node.cls:
                        misclassified = True
                        break

                if misclassified:
                    new_instance = class_instance.ClassificationInstance()
                    for c_e in assigned[c_node.id]:
                        new_instance.add_example(instance.examples[c_e].copy())
                        new_instance.examples[-1].id = len(new_instance.examples)
                    if args.reduce:
                        class_instance.reduce(new_instance)
                    strategy = strat(new_instance)
                    strategy.extend(5)
                    new_tree = eb.run_incremental(encoding, new_instance, Glucose3, strategy, args.time_limit, limits.size_limit, opt_size=args.size, check_mem=True)
                    #new_tree = encoding.run_limited(Glucose3, strategy, limits.size_limit, limits.sample_limit_short, go_up=False, start_bound=len(limits.sample_limit_short)-2, timeout=limits.time_limits[0])

                    if args.reduce:
                        new_instance.unreduce_instance(new_tree)

                    # Append new tree
                    old_id = c_node.id
                    new_tree.root.id = old_id
                    tree.nodes[c_node.id] = decision_tree.DecisionTreeNode(new_tree.root.feature, old_id)

                    # Find parent
                    for x in tree.nodes:
                        if x and not x.is_leaf:
                            if x.left.id == c_node.id:
                                x.left = tree.nodes[c_node.id]
                                break
                            elif x.right.id == c_node.id:
                                x.right = tree.nodes[c_node.id]
                                break

                    add_nodes(new_tree.root)
                    changed = True

    nodes = []
    q = [tree.root]
    while q:
        cn = q.pop()
        nodes.append(cn)

        if not cn.is_leaf:
            q.append(cn.left)
            q.append(cn.right)
else:
    raise RuntimeError("Unsupported mode")

if tree is None:
    print("No tree found.")
    exit(1)

if args.reduce:
    instance.unreduce_instance(tree)
tree.check_consistency()

print(f"END Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
      f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
      f"Time: {time.time() - start_time}")

print(tree.as_string())
