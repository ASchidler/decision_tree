import bdd_instance
import sat_tools
from tree_depth_encoding import TreeDepthEncoding


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


def replace(old_tree, new_tree, root):
    # Clean tree
    q = [root]
    ids = []
    while q:
        c_q = q.pop()

        if not c_q.is_leaf:
            q.append(c_q.left)
            q.append(c_q.right)
            c_q.right = None

        if c_q.id != root.id:
            old_tree.nodes[c_q.id] = None
            c_q.left = None
            ids.append(c_q.id)

    # Add other tree
    root.feature = new_tree.root.feature
    q = [(new_tree.root, root)]
    while q:
        c_q, c_r = q.pop()

        cs = [(c_q.left, True), (c_q.right, False)]
        for cn, cp in cs:
            if cn.is_leaf:
                old_tree.add_leaf(ids.pop(), c_r, cp, cn.cls)
            else:
                n_r = old_tree.add_node(ids.pop(), c_r, c_q.feature, cp)
                q.append((cn, n_r))

    # Sub-tree is now been added in place of the old sub-tree


def leaf_improve(tree, instance, limit=15):
    runner = sat_tools.SatRunner(TreeDepthEncoding, sat_tools.GlucoseSolver(), base_path=".")
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
                c_features.add(c_q.feature)
            else:
                c_leafs.append(c_q.id)

        # Now create a new instance
        feature_map = {}
        c_features = list(c_features)
        for i in range(1, len(c_features)+1):
            feature_map[i] = c_features[i-1]

        new_instance = bdd_instance.BddInstance()
        added = set()
        for s in c_samples:
            values = [None for _ in range(0, len(feature_map) + 1)]
            for k, v in feature_map.items():
                values[k] = instance.examples[s].features[v]

            tp = tuple(values)
            if tp not in added:
                added.add(tp)
                new_instance.add_example(bdd_instance.BddExamples(values, instance.examples[s].cls, len(new_instance.examples)))

        # Solve instance
        new_tree, _ = runner.run(new_instance, max_node[1] - 1, u_bound=max_node[1] - 1)

        # Either the branch is done, or
        if new_tree is None:
            print("Finished sub-tree, no improvement")
            done.update(c_leafs)
        else:
            # Correct features
            q = [new_tree.root]
            while q:
                c_q = q.pop()
                if not c_q.is_leaf:
                    c_q.feature = feature_map[c_q.feature]
                    q.append(c_q.left)
                    q.append(c_q.right)

            # Clean tree
            replace(tree, new_tree, c_parent[0])
            structure = find_structure(tree)
            print("Finished sub-tree, improvement")

