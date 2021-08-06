import nonbinary.decision_tree as decision_tree
from decimal import Decimal


def parse_weka_tree(tree_path, instance, lines=None):
    if lines is None:
        with open(tree_path) as tf:
            lines = []
            for _, l in enumerate(tf):
                lines.append(l)

    wtree = decision_tree.DecisionTree()
    # Single leaf tree, edge case
    if lines[0].strip().startswith(":"):
        pos = lines[0].index(":")
        pos2 = lines[0].index(" ", pos+2)
        cls = lines[0][pos + 2:pos2]
        c_leaf = decision_tree.DecisionTreeLeaf(cls, 1, wtree)
        wtree.nodes[1] = c_leaf
        wtree.root = c_leaf
        return wtree

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
                    pos2 = c_line.index(" ", pos+2)
                    cls = c_line[pos + 2:pos2]
                    wtree.add_leaf(c_id, cp.id, cp.right is not None, cls)
                    c_id += 1

                l_depth = depth
                break
    return wtree


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
            if cstack[-1].left is None:
                cstack[-1].left = cn
            else:
                cstack[-1].right = cn
        cstack.append(cn)

    tree.root = tree.nodes[1]
    return tree
