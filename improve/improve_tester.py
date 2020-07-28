import sys
import parser
import os
import decision_tree
import improve.improver

tree_path = "trees"
instance_path = "experiments/data/full"

instances = [
    "appendicitis",
    "australian",
    "car",
    "haberman",
    "new-thyroid"
]

target_instance = instances[int(sys.argv[1])]
instance = parser.parse(os.path.join(instance_path, target_instance + "-un.csv"))

# Parse tree
with open(os.path.join(tree_path, target_instance+".tree")) as tf:
    lines = []
    for _, l in enumerate(tf):
        lines.append(l)

tree = decision_tree.DecisionTree(instance.num_features, len(lines))
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
                    node = tree.add_node(c_id, cp.id, feature, cp.right is not None)
                else:
                    tree.set_root(feature)
                    node = tree.nodes[1]

            else:
                node = tree.add_leaf(c_id, cp.id, cp.right is not None, c_line.startswith("True"))

            c_id += 1
            l_depth = depth
            stack.append(node)
            break

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