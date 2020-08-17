import sys
import parser
import os
import decision_tree
import improve.improver

tree_path = "trees"
instance_path = "datasets/binary"

instances = [
    "appendicitis",
    "australian",
    "car",
    "haberman",
    "new-thyroid",
    "musk1_bin",
    "mushroom_bin",
    "objectivity_bin",
    "monks-1_bin",
    "ccdefault_bin",
    "hiv_schilling_bin",
    "mammographic_masses_bin",
    "tic-tac-toe_bin"
]


def parse_weka_tree(tree_path):
    with open(tree_path) as tf:
        lines = []
        for _, l in enumerate(tf):
            lines.append(l)

    wtree = decision_tree.DecisionTree(instance.num_features, len(lines) * 2)
    c_id = 1
    l_depth = -1
    stack = []
    for ll in lines:
        depth = 0
        for cc in ll:
            if cc == " " or cc == "|":
                depth += 1
            else:
                c_line = ll[depth:].strip()
                while stack and depth < l_depth:
                    stack.pop()
                    l_depth -= 4

                cp = None if not stack else stack[-1]
                if not c_line.startswith("att"):
                    print(f"Parser error, line should start with att, starts with {c_line}.")
                    exit(1)

                if depth > l_depth:
                    feature = int(c_line[3:c_line.find(" ")])
                    if cp is not None:
                        node = wtree.add_node(c_id, cp.id, feature, cp.right is not None)
                    else:
                        wtree.set_root(feature)
                        node = wtree.nodes[1]
                    stack.append(node)
                    c_id += 1
                    cp = node

                if c_line.find(":") > -1:
                    pos = c_line.index(":")
                    cls = c_line[pos+2:pos+3]
                    wtree.add_leaf(c_id, cp.id, cp.right is not None, cls == "1")
                    c_id += 1

                l_depth = depth
                break
    return wtree


def parse_iti_tree(tree_path):
    with open(tree_path) as tf:
        lines = []
        for _, l in enumerate(tf):
            lines.append(l)

    itree = decision_tree.DecisionTree(instance.num_features, len(lines))
    c_id = 1
    l_depth = -1
    stack = []
    for ll in lines:
        depth = 0
        for cc in ll:
            if cc == " " or cc == "|":
                depth += 1
            else:
                c_line = ll[depth:].strip()
                while stack and depth <= l_depth:
                    stack.pop()
                    l_depth -= 3
                cp = None if not stack else stack[-1]
                if c_line.startswith("att"):
                    feature = int(c_line[3:c_line.find(" ")]) + 1
                    if cp is not None:
                        node = itree.add_node(c_id, cp.id, feature, cp.right is not None)
                    else:
                        itree.set_root(feature)
                        node = itree.nodes[1]

                else:
                    node = itree.add_leaf(c_id, cp.id, cp.right is not None, c_line.startswith("True"))

                c_id += 1
                l_depth = depth
                stack.append(node)
                break
    return itree

target_instance = instances[int(sys.argv[1])]
instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)

tree = parse_weka_tree(os.path.join(tree_path, target_instance+".tree"))
# Parse tree


print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
#
improve.improver.mid_reduced(tree, instance, True)
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")
#
improve.improver.mid_reduced(tree, instance, False)
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")
#
# improve.improver.mid_rearrange(tree, instance, )
# print(f"Tree depth: {tree.get_depth()}")
# print(f"Tree nodes: {tree.get_nodes()}")
# print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")
# #

improve.improver.reduced_leaf(tree, instance)
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")

improve.improver.leaf_rearrange(tree, instance, 10, sample_limit=75)
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")

improve.improver.leaf_select(tree, instance, sample_limit=50)
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
print(f"Tree accuracy: {tree.get_accuracy(instance.examples)}")