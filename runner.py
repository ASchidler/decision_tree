import subprocess
import sys
import os

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


# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (23 * 1024 * 1024 * 1024 // 2, 12 * 1024 * 1024 * 1024))

encodings = [se, da, dp, sn]
strategies = [es.EntropyStrategy, rs.RandomStrategy, ms.MaintainingStrategy]


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
                help="The strategy to use for incremental, recursive mode (0=random, 1=entropy, 2=maintaining)."
                )

args = ap.parse_args()

instance = parser.parse(args.instance, has_header=False)
test_instance = instance
if os.path.exists(args.instance[:-4]+"test"):
    test_instance = parser.parse(args.instance[:-4] + "test")

encoding = encodings[args.encoding]
strat = strategies[args.strategy]

if args.reduce:
    class_instance.reduce(instance)

if args.mode == 0:
    tree = encoding.run(instance, Glucose3, timeout=args.time_limit)
elif args.mode == 1:
    strategy = strat(instance)
    strategy.extend(100)
    tree = encoding.run_incremental(instance, Glucose3, strategy, args.time_limit, limits.size_limit)
elif args.mode == 2:
    strategy = es.EntropyStrategy(instance, stratified=True)
    tree = encoding.run_limited(Glucose3, strategy, limits.size_limit, limits.sample_limit_short)

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

                    strategy = strat(new_instance)
                    new_tree = encoding.run_limited(Glucose3, strategy, limits.size_limit, limits.sample_limit_short, timeout=limits.time_limits[0])

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

print(f"Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
      f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}")

print(tree.as_string())
