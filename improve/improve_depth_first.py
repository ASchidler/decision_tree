import improve.improver as improver


def assign_samples(tree, instance):
    assigned_samples = [[] for _ in tree.nodes]

    for s in instance.examples:
        cnode = tree.root
        assigned_samples[cnode.id].append(s.id - 1)

        while not cnode.is_leaf:
            if s.features[cnode.feature]:
                cnode = cnode.left
            else:
                cnode = cnode.right
            assigned_samples[cnode.id].append(s.id - 1)

    return assigned_samples


def find_deepest_leaf(tree, ignore=None):
    if not ignore:
        ignore = set()

    q = [(0, tree.root)]
    c_max = (-1, None)
    parent = {tree.root.id: None}

    while q:
        c_d, c_n = q.pop()

        if c_n.is_leaf or (c_n.left.id in ignore and c_n.right.id in ignore):
            if c_d > c_max[0]:
                c_max = (c_d, c_n)
        else:
            if c_n.left.id not in ignore:
                parent[c_n.left.id] = c_n
                q.append((c_d+1, c_n.left))
            if c_n.right.id not in ignore:
                parent[c_n.right.id] = c_n
                q.append((c_d+1, c_n.right))

    if c_max[1] is None:
        return -1, None, None

    c_node = [c_max[1]]
    path = []

    while c_node is not None:
        path.append(c_node)
        c_node = parent[c_node.id]

    return c_max[0], c_max[1], path


def run(tree, instance):
    # Select nodes based on the depth
    c_ignore = set()

    while True:
        new_max_d, new_max_n, new_max_p = find_deepest_leaf(tree, c_ignore)

        # No nodes left
        if new_max_n is None:
            break

        assigned = assign_samples(tree, instance)

        # First try to find root
        c_node = new_max_d