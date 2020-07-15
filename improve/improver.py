import sys
from collections import defaultdict

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
    q = [(tree.root, -1, None)]

    while q:
        cnode, d, p = q.pop()
        nodes[cnode.id] = (cnode, d+1, p)

        if not cnode.is_leaf:
            q.append((cnode.left, d+1, cnode.id))
            q.append((cnode.right, d+1, cnode.id))

    return nodes


def depth_from(root):
    d = 0
    q = [(root, 0)]

    while q:
        c_q, c_d = q.pop()

        if c_q.is_leaf:
            d = max(d, c_d)
        else:
            q.append((c_q.left, c_d+1))
            q.append((c_q.right, c_d + 1))

    return d


def replace(old_tree, new_tree, root):
    # Clean tree
    q = [root]
    ids = []
    while q:
        r_n = q.pop()

        if not r_n.is_leaf:
            q.append(r_n.left)
            q.append(r_n.right)
            r_n.right = None
            r_n.left = None

        if r_n.id != root.id:
            old_tree.nodes[r_n.id] = None
            ids.append(r_n.id)

    # Add other tree
    root.feature = new_tree.root.feature
    q = [(new_tree.root, root)]

    while q:
        r_n, r_o = q.pop()

        if not ids:  # Lower max. depth may still have more nodes
            ids.append(len(old_tree.nodes))
            ids.append(len(old_tree.nodes) + 1)
            old_tree.nodes.append(None)
            old_tree.nodes.append(None)

        cs = [(r_n.left, True), (r_n.right, False)]
        for nn, pol in cs:
            if nn.is_leaf:
                old_tree.add_leaf(ids.pop(), r_o.id, pol, nn.cls)
            else:
                n_r = old_tree.add_node(ids.pop(), r_o.id, nn.feature, pol)
                q.append((nn, n_r))

    # Sub-tree is now been added in place of the old sub-tree


def build_unique_set(root, samples, examples, limit=sys.maxsize):
    c_features = set()
    c_leafs = []

    q = [(root, 0)]
    climit = limit
    depth = 0

    while q and climit >= 0:
        c_q, d = q.pop()
        depth = max(depth, d)
        climit -= 1
        if not c_q.is_leaf and climit > 0:
            q.append((c_q.left, d+1))
            q.append((c_q.right, d+1))
            c_features.add(c_q.feature)
        else:
            c_leafs.append(c_q.id)

    feature_map = {}
    c_features = list(c_features)
    for i in range(1, len(c_features) + 1):
        feature_map[i] = c_features[i - 1]

    new_instance = bdd_instance.BddInstance()
    added = set()
    for s in samples:
        values = [None for _ in range(0, len(feature_map) + 1)]
        for k, v in feature_map.items():
            values[k] = examples[s].features[v]

        tp = tuple(values)
        if tp not in added:
            added.add(tp)
            new_instance.add_example(
                bdd_instance.BddExamples(values, examples[s].cls, examples[s].id))

    return new_instance, feature_map, c_leafs, depth


def leaf_rearrange(tree, instance, depth_limit=15, sample_limit=200):
    runner = sat_tools.SatRunner(TreeDepthEncoding, sat_tools.GlucoseSolver(), base_path=".")
    assigned = assign_samples(tree, instance)
    structure = find_structure(tree)

    done = set()

    while True:
        try:
            max_node = max((s for s in structure.values() if s[0].is_leaf and s[0].id not in done), key=lambda x: x[1])
        except ValueError:
            break

        climit = depth_limit
        c_parent = max_node
        c_depth = 0
        last_instance = None
        last_parent = None

        # Find parent of the subtree
        while climit > 0 and c_parent[2] is not None:
            c_parent = structure[c_parent[2]]
            c_instance = build_unique_set(c_parent[0], assigned[c_parent[0].id], instance.examples)
            if len(c_instance[0].examples) > sample_limit:
                break
            last_parent = c_parent
            last_instance = c_instance
            climit -= 1
            c_depth += 1

        if last_instance is None or c_depth <= 1:
            continue

        new_instance, feature_map, c_leafs, _ = last_instance

        # Solve instance
        new_tree, _ = runner.run(new_instance, c_depth - 1, u_bound=c_depth - 1)

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
            replace(tree, new_tree, last_parent[0])
            structure = find_structure(tree)
            assigned = assign_samples(tree, instance)
            print(f"New tree: {new_tree.get_depth()} / {c_depth}")
            print(f"Finished sub-tree, improvement, acc {tree.get_accuracy(instance.examples)}, depth {tree.get_depth()}, root {last_parent[0].id}")


def leaf_select(tree, instance, sample_limit=50, depth_limit=12):
    runner = sat_tools.SatRunner(TreeDepthEncoding, sat_tools.GlucoseSolver(), base_path=".")
    assigned = assign_samples(tree, instance)

    q = [tree.root]
    roots = []

    # Find all subtree roots that assign fewer samples than the limit
    while q:
        c_q = q.pop()
        if not c_q.is_leaf:
            if len(assigned[c_q.id]) <= sample_limit and depth_from(c_q) <= depth_limit:
                roots.append(c_q)
            else:
                q.append(c_q.left)
                q.append(c_q.right)

    for c_r in roots:
        new_instance = bdd_instance.BddInstance()
        for s in assigned[c_r.id]:
            new_instance.add_example(instance.examples[s].copy())

        c_d = depth_from(c_r)
        if c_d > 1:
            new_tree, _ = runner.run(new_instance, c_d-1, u_bound=c_d-1)
            if new_tree is None:
                print(f"Finished sub-tree, no improvement, root {c_r.id}")
            else:
                replace(tree, new_tree, c_r)
                print(
                    f"Finished sub-tree, improvement, acc {tree.get_accuracy(instance.examples)}, depth {tree.get_depth()}, root {c_r.id}")

        # No need to retry, as the number of assigned samples stayed the same


def mid_rearrange(tree, instance, sample_limit=50, depth_limit=12):
    assigned = assign_samples(tree, instance)
    runner = sat_tools.SatRunner(TreeDepthEncoding, sat_tools.GlucoseSolver(), base_path=".")

    for i in range(0, len(tree.nodes)):
        if tree.nodes[i] is None or tree.nodes[i].is_leaf:
            continue

        c_parent = tree.nodes[i]
        last_instance = None

        for r in range(1, depth_limit+1):
            c_instance = build_unique_set(c_parent, assigned[c_parent.id], instance.examples, r)

            if len(c_instance[0].examples) > sample_limit:
                break

            last_instance = c_instance

        if last_instance[3] <= 1:
            continue

        class_mapping = {}
        reverse_mapping = defaultdict(list)
        for cl in last_instance[2]:
            for al in assigned[cl]:
                class_mapping[al] = cl
                reverse_mapping[cl].append(al)

        for ex in last_instance[0].examples:
            ex.cls = class_mapping[ex.id]

        new_tree, _ = runner.run(last_instance[0], last_instance[3] - 1, u_bound=last_instance[3] - 1)

        if new_tree is not None:
            print(f"Found one {new_tree.get_depth()}/{last_instance[3]}, root {c_parent.id}")
        else:
            print(f"Not found {last_instance[3]}, root {c_parent.id}")

    # TODO: Do the same thing with feature reduction?