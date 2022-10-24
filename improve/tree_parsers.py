import decision_tree


def parse_weka_tree(tree_path, instance, lines=None):
    if lines is None:
        with open(tree_path) as tf:
            lines = []
            for _, l in enumerate(tf):
                lines.append(l)

    wtree = decision_tree.DecisionTree(instance.num_features, len(lines) * 2)
    # Single leaf tree, edge case
    if lines[0].strip().startswith(":"):
        pos = lines[0].index(":")
        pos2 = lines[0].index(" ", pos+2)
        cls = lines[0][pos + 2:pos2]
        c_leaf = decision_tree.DecisionTreeLeaf(cls, 1)
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
        if ll.startswith("Pruning tree"):
            continue

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
                    feature = int(c_line[3:c_line.find(" ")]) #+ 1
                    if cp is not None:
                        node = itree.add_node(c_id, cp.id, feature, cp.right is not None)
                    else:
                        itree.set_root(feature)
                        node = itree.nodes[1]

                else:
                    # Add leaf
                    classes = [x.strip().split(" ") for x in c_line.split(")") if len(x.strip()) > 0]
                    # Distinguish between pruned (more than one class per leaf) or unpruned
                    if len(classes) == 1:
                        c_cls = classes[0][0]
                    else:
                        c_cls, _ = max(classes, key=lambda x: int(x[1][1:])) # 1: to skip leading (

                    if cp is not None:
                        node = itree.add_leaf(c_id, cp.id, cp.right is not None, c_cls)
                    else:
                        node = decision_tree.DecisionTreeLeaf(c_cls, c_id)
                        itree.nodes[1] = node
                        itree.root = node

                c_id += 1
                l_depth = depth
                stack.append(node)
                break
    return itree


def parse_internal_tree(tree_path, instance):
    with open(tree_path) as tf:
        lines = []
        for _, cl in enumerate(tf):
            cl = cl.strip()
            if len(cl) > 0:
                lines.append(cl)

    if len(lines) == 0:
        return None

    tree = decision_tree.DecisionTree(instance.num_features, len(lines))

    cstack = []
    id = 0

    for cl in lines:
        id += 1
        cf = cl.split("-")
        depth = len(cf) - 1

        if cf[-1].startswith("a"):
            cn = decision_tree.DecisionTreeNode(int(cf[-1].strip()[2:-1]), id)
        else:
            cc = cf[-1].strip()[2:-1]
            cn = decision_tree.DecisionTreeLeaf(cc, id)
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
