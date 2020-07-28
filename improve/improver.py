import sys
from collections import defaultdict

import bdd_instance
import sat_tools
from tree_depth_encoding import TreeDepthEncoding
from decision_tree import DecisionTreeNode, DecisionTreeLeaf
import heapq


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

        if len(ids) < 2:  # Lower max. depth may still have more nodes
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


def stitch(old_tree, new_tree, root):
    # Remove unnecessary

    # find leaves in new node
    q = [new_tree.get_root()]
    leaves = []
    hit_count = defaultdict(list)
    while q:
        c_q = q.pop()
        if c_q.is_leaf:
            if c_q.cls >= 0:
                leaves.append(c_q)
                hit_count[c_q.cls].append(c_q)
        else:
            q.extend(c_q.get_children().values())

    original_ids = set(x.cls for x in leaves)

    # Duplicate structures for leaves used multiple times
    for k, v in hit_count.items():
        if len(v) > 1:
            # copy
            for c_l in v[1:]:
                n_id = len(old_tree.nodes)
                if old_tree.nodes[k].is_leaf:
                    old_tree.nodes.append(DecisionTreeLeaf(old_tree.nodes[k].cls, n_id))
                else:
                    old_tree.nodes.append(DecisionTreeNode(old_tree.nodes[k].feature, n_id))
                c_l.cls = n_id

                if not old_tree.nodes[k].is_leaf:
                    q = [(old_tree.nodes[k], old_tree.nodes[-1])]

                    while q:
                        o_r, n_r = q.pop()
                        for c_c, c_p in [(o_r.left, True), (o_r.right, False)]:
                            n_id = len(old_tree.nodes)
                            old_tree.nodes.append(None)

                            if c_c.is_leaf:
                                old_tree.add_leaf(n_id, n_r.id, c_p, c_c.cls)
                            else:
                                nn = old_tree.add_node(n_id, n_r.id, c_c.feature, c_p)
                                q.append((c_c, nn))

    # Eliminate old nodes
    ids = []
    q = [root]
    while q:
        c_q = q.pop()

        if c_q.id not in original_ids:
            if not c_q.is_leaf:
                q.append(c_q.left)
                q.append(c_q.right)
            if c_q.id != root.id:
                old_tree.nodes[c_q.id] = None
                ids.append(c_q.id)

    # Stitch in new tree
    q = [(root, new_tree.get_root())]
    root.feature = new_tree.get_root().feature
    root.left = None
    root.right = None

    while q:
        o_r, n_r = q.pop()

        if len(ids) < 2:  # Lower max. depth may still have more nodes
            ids.append(len(old_tree.nodes))
            ids.append(len(old_tree.nodes) + 1)
            old_tree.nodes.append(None)
            old_tree.nodes.append(None)

        for c_p, c_c in [(True, n_r.get_children()[True]), (False, n_r.get_children()[False])]:
            if c_c.is_leaf:
                if c_c.cls < 0:
                    old_tree.add_leaf(ids.pop(), o_r.id, c_p, True if c_c.cls == -2 else False)
                else:
                    if c_p:
                        o_r.left = old_tree.nodes[c_c.cls]
                    else:
                        o_r.right = old_tree.nodes[c_c.cls]
            else:
                n_r = old_tree.add_node(ids.pop(), o_r.id, c_c.feature, c_p)
                q.append((n_r, c_c))


def build_unique_set(root, samples, examples, limit=sys.maxsize):
    c_features = set()
    c_leafs = []

    q = [(root, 0)]
    depth = 0

    while q:
        c_q, d = q.pop()
        depth = max(depth, d)

        if c_q.is_leaf or d >= limit:
            c_leafs.append(c_q.id)
        else:
            q.append((c_q.left, d + 1))
            q.append((c_q.right, d + 1))
            c_features.add(c_q.feature)

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


def build_reduced_set(root, tree, examples, assigned, depth_limit, sample_limit, reduce):
    max_dist = depth_from(root)
    q = [[] for _ in range(0, max_dist+1)]
    q[max_dist].append((0, root))

    features = set()
    last_instance = None
    cnt = 0
    frontier = {root.id}
    max_depth = 0

    while q:
        while not q[-1]:
            q.pop()

        if not q:
            break

        new_nodes = q.pop()
        for c_depth, new_root in new_nodes:
            c_max_depth = max(max_depth, c_depth + 1)
            cnt += 1

            if not new_root.is_leaf:
                features.add(new_root.feature)
                q[depth_from(new_root.left)].append((c_depth + 1, new_root.left))
                q[depth_from(new_root.right)].append((c_depth + 1, new_root.right))

                frontier.remove(new_root.id)
                frontier.add(new_root.left.id)
                frontier.add(new_root.right.id)

        if c_max_depth > depth_limit:
            break

        if cnt >= 3:
            class_mapping = {}
            cnt_internal = 0
            for c_leaf in frontier:
                for s in assigned[c_leaf]:
                    if tree.nodes[c_leaf].is_leaf:
                        class_mapping[s] = -2 if tree.nodes[c_leaf].cls else -1
                    else:
                        cnt_internal += 1
                        class_mapping[s] = c_leaf

            # If all "leaves" are leaves, this method is not required, as it will be handled by separate improvements
            if cnt_internal > 0:
                new_instance = bdd_instance.BddInstance()
                for s in assigned[root.id]:
                    new_instance.add_example(bdd_instance.BddExamples(examples[s].features, class_mapping[s], examples[s].id))

                if reduce:
                    key = new_instance.min_key(randomize=True)
                    bdd_instance.reduce(new_instance, min_key=key)
                else:
                    bdd_instance.reduce(new_instance, min_key=features)

                # TODO: This leads to adding as many nodes as possible. To emphasize the remaining depth more,
                #  one should stop when the node with the highest remaining depth fails due to too high depth
                #  or too many samples
                if len(new_instance.examples) <= sample_limit:
                    last_instance = new_instance
                    max_depth = c_max_depth
                else:
                    q = None

    return last_instance, max_depth


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
            print(f"Finished sub-tree, no improvement {c_depth}")
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
        for cl in last_instance[2]:
            for al in assigned[cl]:
                class_mapping[al + 1] = cl

        for ex in last_instance[0].examples:
            ex.cls = class_mapping[ex.id]

        new_tree, _ = runner.run(last_instance[0], last_instance[3] - 1, u_bound=last_instance[3] - 1)

        if new_tree is not None:
            q = [new_tree.nodes[0]]
            while q:
                c_q = q.pop()
                if not c_q.is_leaf:
                    c_q.feature = last_instance[1][c_q.feature]
                    q.append(c_q.children[True])
                    q.append(c_q.children[False])
            # Stitch the new tree in the middle
            stitch(tree, new_tree, c_parent)
            print(
                f"Found one {new_tree.get_depth()}/{last_instance[3]}, root {c_parent.id}, acc {tree.get_accuracy(instance.examples)}")
            assigned = assign_samples(tree, instance)
        else:
            print(f"Not found {last_instance[3]}, root {c_parent.id}")

    # TODO: Do the same thing with feature reduction?


def reduced_leaf(tree, instance, sample_limit=50, depth_limit=15):
    assigned = assign_samples(tree, instance)
    runner = sat_tools.SatRunner(TreeDepthEncoding, sat_tools.GlucoseSolver(), base_path=".")

    for i in range(0, len(tree.nodes)):
        if tree.nodes[i] is None or tree.nodes[i].is_leaf or depth_from(tree.nodes[i]) > depth_limit:
            continue

        new_instance = bdd_instance.BddInstance()
        for s in assigned[i]:
            new_instance.add_example(instance.examples[s].copy())

        bdd_instance.reduce(new_instance)

        if len(new_instance.examples) <= sample_limit:
            nd = depth_from(tree.nodes[i])
            new_tree, _ = runner.run(new_instance,  nd-1, u_bound=nd - 1)

            if new_tree is not None:
                new_instance.unreduce_instance(new_tree)
                replace(tree, new_tree, tree.nodes[i])
                assigned = assign_samples(tree, instance)
                print(
                    f"Finished sub-tree, improvement, acc {tree.get_accuracy(instance.examples)}, depth {tree.get_depth()}, root {tree.nodes[i].id}")
            else:
                print("No improvement")
        else:
            print("Too big")


def mid_reduced(tree, in_instance, reduce, sample_limit=50, depth_limit=12):
    assigned = assign_samples(tree, in_instance)
    runner = sat_tools.SatRunner(TreeDepthEncoding, sat_tools.GlucoseSolver(), base_path=".")

    for i in range(0, len(tree.nodes)):
        # Exclude nodes with fewer than limit samples, as this will be handled by the leaf methods
        if tree.nodes[i] is None or tree.nodes[i].is_leaf or len(assigned[tree.nodes[i].id]) < sample_limit:
            continue

        c_parent = tree.nodes[i]
        instance, i_depth = build_reduced_set(c_parent, tree, in_instance.examples, assigned, depth_limit, sample_limit, reduce)

        if instance is None:
            continue

        new_tree, _ = runner.run(instance, i_depth - 1, u_bound=i_depth-1)

        if new_tree is not None:
            instance.unreduce_instance(new_tree)

            # Stitch the new tree in the middle
            stitch(tree, new_tree, c_parent)
            print(
                f"Found one {new_tree.get_depth()}/{i_depth}, root {c_parent.id}, acc {tree.get_accuracy(in_instance.examples)}")
            assigned = assign_samples(tree, in_instance)
        else:
            print(f"Not found {i_depth}, root {c_parent.id}")