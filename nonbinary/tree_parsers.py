import nonbinary.decision_tree as decision_tree
from decimal import Decimal


def parse_internal_tree(tree_path):
    with open(tree_path) as tf:
        lines = []
        for _, cl in enumerate(tf):
            cl = cl.strip()
            if len(cl) > 0:
                lines.append(cl)

    if len(lines) == 0:
        return None

    tree = decision_tree.DecisionTree()

    cstack = []
    id = 0

    for cl in lines:
        id += 1
        end_idx = 0
        for i in range(0, len(cl)):
            if cl[i] != "-":
                end_idx = i
                break

        depth = end_idx
        tree.nodes.append(None)
        cf = cl[end_idx:]
        if cf.startswith("a"):
            a_fields = cf.split(" ")
            is_categorical = a_fields[1] == "="
            cn = decision_tree.DecisionTreeNode(int(a_fields[0].strip()[2:-1]), a_fields[2] if is_categorical else Decimal(a_fields[2]), id, decision_tree, is_categorical)
        else:
            cc = cf.strip()[2:-1]
            cn = decision_tree.DecisionTreeLeaf(cc, id, decision_tree)
        tree.nodes[id] = cn

        while depth < len(cstack):
            cstack.pop()

        # root
        if depth == 0:
            pass
        else:
            cn.parent = cstack[-1]
            if cstack[-1].left is None:
                cstack[-1].left = cn
            else:
                cstack[-1].right = cn
        cstack.append(cn)

    tree.root = tree.nodes[1]
    return tree
