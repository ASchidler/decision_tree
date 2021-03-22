import sys
from collections import defaultdict

import class_instance
from pysat.solvers import Glucose3
from decision_tree import DecisionTreeNode, DecisionTreeLeaf
from sat import switching_encoding, depth_avellaneda, depth_partition


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
            if c_q.cls >= 0 and c_q.cls != False and c_q.cls != True:
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

    new_instance = class_instance.ClassificationInstance()
    added = set()
    for s in samples:
        values = [None for _ in range(0, len(feature_map) + 1)]
        for k, v in feature_map.items():
            values[k] = examples[s].features[v]

        tp = tuple(values)
        if tp not in added:
            added.add(tp)
            new_instance.add_example(
                class_instance.ClassificationExample(values, examples[s].cls, examples[s].id))

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

            if not new_root.is_leaf and new_root.id in frontier:
                features.add(new_root.feature)
                q[depth_from(new_root.left)].append((c_depth + 1, new_root.left))
                q[depth_from(new_root.right)].append((c_depth + 1, new_root.right))

                frontier.remove(new_root.id)
                frontier.add(new_root.left.id)
                frontier.add(new_root.right.id)

        # Complete with leafs...
        for cl in list(frontier):
            c_n = tree.nodes[cl]
            if not c_n.is_leaf and c_n.left.is_leaf and c_n.right.is_leaf:
                frontier.remove(cl)
                frontier.add(c_n.left.id)
                frontier.add(c_n.right.id)

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
                new_instance = class_instance.ClassificationInstance()
                for s in assigned[root.id]:
                    new_instance.add_example(class_instance.ClassificationExample(examples[s].features, class_mapping[s], examples[s].id))

                if reduce:
                    # key = new_instance.min_key(randomize=True)
                    class_instance.reduce(new_instance, randomized_runs=1)
                else:
                    class_instance.reduce(new_instance, min_key=features)

                # TODO: This leads to adding as many nodes as possible. To emphasize the remaining depth more,
                #  one should stop when the node with the highest remaining depth fails due to too high depth
                #  or too many samples
                if len(new_instance.examples) <= sample_limit[c_max_depth]:
                    last_instance = new_instance
                    max_depth = c_max_depth
                else:
                    q = None

    return last_instance, max_depth


def build_runner():
    #enc = switching_encoding.SwitchingEncoding()
    #enc = depth_avellaneda.DepthAvellaneda()
    enc = depth_partition.DepthPartition()
    return lambda i, b, t, ub: enc.run(i, Glucose3, start_bound=b, timeout=t, ub=ub)


def leaf_rearrange(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, tmp_dir="."):
    runner = build_runner()

    prev_instance = None
    prev_idx = path_idx

    while path_idx < len(path):
        c_d = depth_from(path[path_idx])
        if c_d > depth_limit:
            break
        new_instance = build_unique_set(path[path_idx], assigned[path[path_idx].id], instance.examples)
        if len(new_instance[0].examples) > sample_limit[c_d]:
            break

        prev_instance = new_instance
        prev_idx = path_idx
        path_idx += 1

    if prev_instance is not None and len(prev_instance[0].examples) > 0:
        node = path[prev_idx]
        new_instance, feature_map, c_leafs, _ = prev_instance
        cd = depth_from(node)

        # Solve instance
        new_tree = runner(new_instance, cd - 1, time_limit, ub=cd-1)

        # Either the branch is done, or
        if new_tree is None:
            return False, prev_idx
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
            replace(tree, new_tree, node)
            return True, prev_idx

    return False, prev_idx


def leaf_select(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, tmp_dir="."):
    last_idx = path_idx
    while path_idx < len(path):
        c_d = depth_from(path[path_idx])
        if c_d > depth_limit or len(assigned[path[path_idx].id]) > sample_limit[c_d]:
            break
        last_idx = path_idx
        path_idx += 1

    node = path[last_idx]
    c_d = depth_from(node)
    if not (2 < c_d <= depth_limit) or len(assigned[node.id]) > sample_limit[c_d]:
        return False, last_idx

    runner = build_runner()

    new_instance = class_instance.ClassificationInstance()
    for s in assigned[node.id]:
        new_instance.add_example(instance.examples[s].copy())

    if len(new_instance.examples) == 0:
        return False, last_idx

    new_tree = runner(new_instance, c_d-1, time_limit, ub=c_d-1)
    if new_tree is None:
        return False, last_idx
    else:
        replace(tree, new_tree, node)
        return True, last_idx


def mid_rearrange(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, tmp_dir="."):
    runner = build_runner()

    if path[path_idx].is_leaf:
        return False, path_idx

    c_parent = path[path_idx]
    last_instance = None

    for r in range(1, depth_limit+1):
        c_instance = build_unique_set(c_parent, assigned[c_parent.id], instance.examples, r)

        if len(c_instance[0].examples) > sample_limit[c_instance[3]]:
            break

        last_instance = c_instance

    if last_instance[3] <= 1:
        return False, path_idx

    class_mapping = {}
    for cl in last_instance[2]:
        for al in assigned[cl]:
            class_mapping[al + 1] = cl

    for ex in last_instance[0].examples:
        ex.cls = class_mapping[ex.id]

    if len(last_instance[0].examples) == 0:
        return False, path_idx

    new_tree = runner(last_instance[0], last_instance[3] - 1, time_limit, ub=last_instance[3]-1)

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
        return True, path_idx

    return False, path_idx


def reduced_leaf(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, tmp_dir="."):
    runner = build_runner()

    prev_instance = None
    prev_idx = path_idx

    while True:
        if path_idx >= len(path):
            break
        c_d = depth_from(path[path_idx])
        if c_d > depth_limit:
            break

        new_instance = class_instance.ClassificationInstance()
        for s in assigned[path[path_idx].id]:
            new_instance.add_example(instance.examples[s].copy())

        if len(new_instance.examples) == 0:
            break

        instance.reduce(new_instance, randomized_runs=1)

        if len(new_instance.examples) > sample_limit[c_d]:
            break

        prev_instance = new_instance
        prev_idx = path_idx
        path_idx += 1

    if prev_instance is not None:
        node = path[prev_idx]
        nd = depth_from(node)
        new_tree = runner(prev_instance,  nd-1, time_limit, ub=nd-1)

        if new_tree is not None:
            prev_instance.unreduce_instance(new_tree)
            replace(tree, new_tree, node)

            return True, prev_idx

    return False, prev_idx


def mid_reduced(tree, instance, path_idx, path, assigned, reduce, sample_limit, depth_limit, time_limit, tmp_dir="."):
    runner = build_runner()

    # Exclude nodes with fewer than limit samples, as this will be handled by the leaf methods
    if path[path_idx].is_leaf:
        return False, path_idx

    c_parent = path[path_idx]
    instance, i_depth = build_reduced_set(c_parent, tree, instance.examples, assigned, depth_limit, sample_limit, reduce)

    if instance is None or len(instance.examples) == 0:
        return False, path_idx

    new_tree = runner(instance, i_depth - 1, time_limit, ub=i_depth-1)

    if new_tree is not None:
        instance.unreduce_instance(new_tree)

        # Stitch the new tree in the middle
        stitch(tree, new_tree, c_parent)
        return True, path_idx

    return False, path_idx
