import nonbinary.decision_tree as decision_tree
from decimal import Decimal, InvalidOperation


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

    for cl in lines:
        end_idx = 0
        for i in range(0, len(cl)):
            if cl[i] != "-":
                end_idx = i
                break

        cf = cl[end_idx:]

        if cf.startswith("a"):
            a_fields = cf.split(" ")
            is_categorical = a_fields[1].strip() == "="
            c_f = int(a_fields[0].strip()[2:-1])
            fd = a_fields[2] if is_categorical else Decimal(a_fields[2])

            try:
                fd = int(fd)
            except ValueError:
                try:
                    decimals = len(fd.split(".")[-1])
                    if decimals <= 6:
                        fd = Decimal(fd)
                    else:
                        # Truncate after 6 decimals, since weka does this
                        Decimal(fd)
                        fd = Decimal(fd[:-(decimals - 6)])
                except InvalidOperation:
                    try:
                        # Special case to catch scientific notation
                        fd = float(fd)
                        # Round to 6 decimals
                        fd = Decimal(int(fd * 1000000) / 1000000.0)
                    except ValueError:
                        pass
            if not cstack:
                n_n = tree.set_root(c_f, fd, is_categorical)
            else:
                n_n = tree.add_node(c_f, fd, cstack[-1].id, cstack[-1].left is None, is_categorical)
            cstack.append(n_n)
        else:
            cc = cf.strip()[2:-1]
            if not cstack:
                tree.set_root_leaf(cc)
            else:
                tree.add_leaf(cc, cstack[-1].id, cstack[-1].left is None)

        while cstack and cstack[-1].right is not None:
            cstack.pop()

    tree.root = tree.nodes[1]
    return tree
