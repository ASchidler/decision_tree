import sys
import parser
import os
import decision_tree
import improve.improve_depth_first as df

tree_path = "datasets/trees"
instance_path = "datasets/split"

target_instance = sys.argv[1]


def parse_weka_tree(tree_path, instance):
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


def parse_iti_tree(tree_path, instance):
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


training_instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)
test_instance = parser.parse(os.path.join(instance_path, target_instance + ".test"), has_header=False)

tree = parse_weka_tree(os.path.join(tree_path, target_instance+".tree"), training_instance)
# Parse tree


print(f"Tree training accuracy: {tree.get_accuracy(training_instance.examples)}")
print(f"Tree test accuracy: {tree.get_accuracy(test_instance.examples)}")
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
#
df.run(tree, training_instance, test_instance)

print(f"Tree training accuracy: {tree.get_accuracy(training_instance.examples)}")
print(f"Tree test accuracy: {tree.get_accuracy(test_instance.examples)}")
print(f"Tree depth: {tree.get_depth()}")
print(f"Tree nodes: {tree.get_nodes()}")
