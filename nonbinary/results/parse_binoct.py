import os.path
import sys
from nonbinary.nonbinary_instance import parse
from nonbinary.decision_tree import DecisionTree
import decimal
from collections import defaultdict

target = sys.argv[1]
pth = "nonbinary/instances"
out_path = "nonbinary/results/trees/binoct"
result_path = "nonbinary/results/results_binoct.csv"

results = defaultdict(list)


def parse_tree(data, fl, slice, target_depth):
    instance, instance_test, instance_validation = parse(pth, fl, int(slice))
    feature_maps = {}
    for c_f in instance.is_categorical:
        feature_maps[c_f] = {i + 1: x for i, x in enumerate(sorted(instance.domains[c_f]))}
    #cls_map = {i + 1: x for i, x in enumerate(sorted(instance.classes))}
    tree = DecisionTree()

    def change_level(p, idx):
        leaf_done = False
        n_n = None
        while idx < len(data):
            ln = data[idx].strip()
            if ln.startswith("if"):
                paranth = ln[ln.index("(")+1:ln.index(")")-1]
                feature, thresh = paranth.split("<=")
                is_cat = False

                if feature.find(":") > -1:
                    feature, thresh_idx = feature.split(":")
                    feature = int(feature.strip())
                    thresh = feature_maps[feature][int(thresh_idx.strip())]
                    is_cat = True
                else:
                    feature = int(feature.strip())
                    thresh = decimal.Decimal(thresh.strip())

                if p is None:
                    n_n = tree.set_root(feature, thresh, is_cat)
                else:
                    if p.left is not None and p.right is not None:
                        print("Error1")
                    n_n = tree.add_node(feature, thresh, p.id, p.left is None, is_cat)
                idx = change_level(n_n, idx+1)
            elif ln.startswith("else {"):
                idx = change_level(n_n, idx+1)
            elif ln.startswith("return "):
                if not leaf_done:
                    leaf_done = True
                    if p is None:
                        tree.set_root_leaf("x")
                    else:
                        if p.left is not None and p.right is not None:
                            print("Error2")
                        tree.add_leaf(next(iter(instance.classes)), p.id, p.left is None)
                idx += 1
            elif ln.startswith("}"):
                return idx + 1
            else:
                idx += 1

    change_level(None, 0)
    tree.root.reclassify(instance.examples)
    results[fl].append((tree.get_nodes(), tree.get_depth(), tree.get_accuracy(instance.examples),
                        tree.get_accuracy(instance_test.examples), tree.get_avg_length(instance_test.examples)))
    print(f"{fl}{slice} {tree.get_depth()} {tree.get_nodes()} {tree.get_accuracy(instance_test.examples)}")
    with open(os.path.join(out_path, f"{fl}.{slice}.{target_depth}.dt"), "w") as outp:
        outp.write(tree.as_string())

    tree.as_string()
    return tree



target_depth = target.split(".")[1]

with open(target) as inp:
    tree_data = []
    tree_mode = False
    c_file = None
    c_slice = None
    for cl in inp:
        if cl.startswith("[('"):
            c_file = cl.strip().split(" ")[-1]
            c_file = c_file.split("/")[1]
            c_file, c_slice, _ = c_file.split(".")
            c_slice = c_slice.replace("train", "")

        if cl.startswith("if"):
            tree_mode = True

        if tree_mode:
            tree_data.append(cl.strip())

        if cl.startswith("num") and tree_mode:
            parse_tree(tree_data, c_file, c_slice, target_depth)
            tree_data = []
            tree_mode = False
            c_file = None
            c_slice = None

