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

random.seed = 1

tree_path = "datasets/trees"
instance_path = "datasets/split"
tmp_dir = "."
is_iti = False
is_size = False

i = 2
while i < len(sys.argv):
    if sys.argv[i] == "-i":
        is_iti = True
    elif sys.argv[i] == "-s":
        is_size = True
    elif sys.argv[i] == "-t":
        tmp_dir = sys.argv[i+1]
        i += 1
    else:
        print(f"Unknown argument {sys.argv[i]}")
    i += 1



fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
fls.sort()
target_instance_idx = int(sys.argv[1])

if target_instance_idx > len(fls):
    print(f"Only {len(fls)} files are known.")
    exit(1)

target_instance = fls[target_instance_idx-1][:-5]

training_instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)
test_instance = parser.parse(os.path.join(instance_path, target_instance + ".test"), has_header=False)

if is_iti:
    tree = parse_iti_tree(os.path.join(tree_path, target_instance+".iti"), training_instance)
else:
    tree = parse_weka_tree(os.path.join(tree_path, target_instance+".tree"), training_instance)
# Parse tree

print(f"{target_instance}: Features {training_instance.num_features}\tExamples {len(training_instance.examples)}\t"
      f"Optimize {'Depth' if not is_size else 'Size'}\tHeuristic {'Weka' if not is_iti else 'ITI'}")

print(f"Time: Start\t\t"
      f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
      f"Depth {tree.get_depth():03}\t"
      f"Avg {tree.get_avg_depth():03.4f}\t"
      f"Nodes {tree.get_nodes()}")

if is_size:
    sf.run(tree, training_instance, test_instance, tmp_dir=tmp_dir)
else:
    df.run(tree, training_instance, test_instance, tmp_dir=tmp_dir)

print(f"Time: End\t\t"
      f"Training {tree.get_accuracy(training_instance.examples):.4f}\t"
      f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
      f"Depth {tree.get_depth():03}\t"
      f"Avg {tree.get_avg_depth():03.4f}\t"
      f"Nodes {tree.get_nodes()}")
