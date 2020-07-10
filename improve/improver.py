import bdd_instance

def assign_samples(tree, instance):
    assigned_samples = [[] for _ in tree.nodes]

    for s in instance.examples:
        cnode = tree.root
        assigned_samples[cnode.id].append(s.id)

        while not cnode.is_leaf:
            if s.features[cnode.feature]:
                cnode = cnode.left
            else:
                cnode = cnode.right
            assigned_samples[cnode.id].append(s.id)

    return assigned_samples


def find_structure(tree):
    nodes = {}
    q = [(tree.root, 0, None)]

    while q:
        cnode, d, p = q.pop()
        nodes[cnode.id] = (cnode, d+1, p)

        if not cnode.is_leaf:
            q.append((cnode.left, d+1, cnode.id))
            q.append((cnode.right, d+1, cnode.id))

    return nodes


def leaf_improve(tree, instance, limit=15):
    assigned = assign_samples(tree, instance)
    structure = find_structure(tree)

    done = set()

    while True:
        try:
            max_node = max((s for s in structure.values() if s[0].is_leaf and s[0].id not in done), key=lambda x: x[1])
        except ValueError:
            break

        climit = limit
        c_parent = max_node

        # Find parent of the subtree
        while climit > 0 and c_parent[2] is not None:
            c_parent = structure[c_parent[2]]
            climit -= 1

        # Now traverse and find involved leaves as well as features
        c_samples = assigned[c_parent[0].id]
        c_features = set()
        c_leafs = []

        q = [c_parent[0]]
        while q:
            c_q = q.pop()
            if not c_q.is_leaf:
                q.append(c_q.left)
                q.append(c_q.right)
                c_features.add(c_q[0].feature)
            else:
                c_leafs.append(c_q[0].id)

        # Now create a new instance
        feature_map = {}
        c_features = list(c_features)
        for i in range(0, len(c_features)):
            feature_map[i] = c_features[i]

        new_instance = bdd_instance.BddInstance()
        for s in c_samples[c_parent[0].id]:
            values = [None for _ in range(0, len(feature_map))]
            for k, v in feature_map.items():
                values[k] = instance.examples[s].examples[v]

            tp = tuple(values)
            if tp not in
            new_instance.add_example()