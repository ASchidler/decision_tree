from sys import maxsize
from nonbinary.nonbinary_instance import ClassificationInstance, Example
from decision_tree import DecisionTreeLeaf, DecisionTreeNode
from collections import defaultdict
import nonbinary.depth_avellaneda_base as bs

literal_limit = 200 * 1000 * 1000


def build_unique_set(root, samples, examples, limit=maxsize):
    c_features = set()
    c_leafs = []

    q = [(root, 0)]
    depth = 0

    # Find all features used in the tree
    while q:
        c_q, d = q.pop()
        depth = max(depth, d)

        if c_q.is_leaf or d >= limit:
            c_leafs.append(c_q.id)
        else:
            q.append((c_q.left, d + 1))
            q.append((c_q.right, d + 1))
            c_features.add(c_q.feature)

    # We need a guaranteed order
    c_features = list(c_features)

    new_instance = ClassificationInstance()
    added = {}
    for s in samples:
        tp = tuple(s.features[v] for v in c_features)

        if tp not in added:
            added[tp] = s.cls
            new_instance.add_example(Example(new_instance, list(tp), f"-{s.cls}"))
        else:
            # This check is necessary: if the decision tree is pruned, some samples may be misclassified.
            # In that case, the feature set of the sub-tree may not be a support set.
            if added[tp] != s.cls:
                return None
    new_instance.finish()
    return new_instance, c_features, c_leafs, depth


def build_reduced_set(root, tree, examples, assigned, depth_limit, sample_limit, reduce, encoding):
    max_dist = root.get_depth()
    q = [[] for _ in range(0, max_dist+1)]
    q[max_dist].append((0, root))

    features = set()
    last_instance = None
    cnt = 0
    frontier = {root.id}
    max_depth = 0
    c_max_depth = 0

    while q:
        while q and not q[-1]:
            q.pop()

        if not q:
            break

        new_nodes = q.pop()
        for c_depth, new_root in new_nodes:
            c_max_depth = max(max_depth, c_depth + 1)
            cnt += 1

            if not new_root.is_leaf and new_root.id in frontier:
                features.add(new_root.feature)
                q[new_root.left.get_depth()].append((c_depth + 1, new_root.left))
                q[new_root.right.get_depth()].append((c_depth + 1, new_root.right))

                frontier.remove(new_root.id)
                frontier.add(new_root.left.id)
                frontier.add(new_root.right.id)

        # Complete with leafs, if possible, as they won't add any samples
        for cl in list(frontier):
            c_n = tree.nodes[cl]
            if not c_n.is_leaf and c_n.left.is_leaf and c_n.right.is_leaf:
                frontier.remove(cl)
                features.add(c_n.feature)
                frontier.add(c_n.left.id)
                frontier.add(c_n.right.id)

        if c_max_depth > depth_limit:
            break

        if cnt >= 3:
            class_sizes = {}
            class_mapping = {}
            cnt_internal = 0
            for c_leaf in frontier:
                for s in assigned[c_leaf]:
                    if tree.nodes[c_leaf].is_leaf:
                        class_mapping[s.id] = f"-{tree.nodes[c_leaf].cls}"
                    else:
                        cnt_internal += 1
                        class_mapping[s.id] = f"{c_leaf}"
                        class_sizes[f"{c_leaf}"] = tree.nodes[c_leaf].get_leaves()


            # If all "leaves" are leaves, this method is not required, as it will be handled by separate improvements
            if cnt_internal > 0:
                new_instance = ClassificationInstance()
                for s in assigned[root.id]:
                    n_s = s.copy(new_instance)
                    n_s.cls = class_mapping[s.id]
                    if not n_s.cls.startswith("-"):
                        n_s.surrogate_cls = f"-{s.cls}"
                    new_instance.add_example(n_s)
                new_instance.class_sizes = class_sizes
                new_instance.finish()
                if reduce:
                    new_instance.reduce_with_key()
                else:
                    new_instance.reduce(features)

                # TODO: This leads to adding as many nodes as possible. To emphasize the remaining depth more,
                #  one should stop when the node with the highest remaining depth fails due to too high depth
                #  or too many samples
                if encoding.estimate_size(new_instance, c_max_depth-1) <= literal_limit:
                    if len(new_instance.examples) <= sample_limit[c_max_depth] \
                            or (last_instance is None and len(new_instance.examples) < sample_limit[1]):
                        last_instance = new_instance
                        max_depth = c_max_depth
                else:
                    q = None

    return last_instance, max_depth


def stitch(old_tree, new_tree, root, instance):
    # Remove unnecessary

    # find leaves that are used in new tree
    leaves = set()
    q = [new_tree.root]

    while q:
        c_q = q.pop()
        if c_q.is_leaf:
            if not c_q.cls.startswith("-"):
                leaves.add(int(c_q.cls))
        else:
            q.extend(c_q.get_children())

    # Eliminate old nodes
    old_tree.c_idx = 1
    q = [root]

    while q:
        c_q = q.pop()

        # Stop when we hit one of the "leaves", this is the target to stitch
        if c_q.id not in leaves:
            if not c_q.is_leaf:
                q.extend(c_q.get_children())
                c_q.left = None
                c_q.right = None
            if c_q.id != root.id:
                old_tree.nodes[c_q.id] = None

    def duplicate(new_parent, old_root_id, pol):
        """Duplicates the sub tree rooted at old_root under new_parent"""

        s_q = [(old_tree.nodes[old_root_id], new_parent, pol)]

        while s_q:
            c_child, c_parent, c_pol = s_q.pop()
            if c_child.is_leaf:
                old_tree.add_leaf(c_child.cls, c_parent.id, c_pol)
            else:
                new_child = old_tree.add_node(c_child.feature, c_child.threshold, c_parent.id, c_pol, c_child.is_categorical)
                s_q.append((c_child.left, new_child, True))
                s_q.append((c_child.right, new_child, False))

    used_leaves = set()
    duplicated = False

    # Stitch in new tree
    if new_tree.root.is_leaf:
        is_left = root.parent.left.id == root.id
        old_tree.nodes[root.id] = None
        if root.id != old_tree.root.id:
            old_tree.add_leaf(new_tree.root.cls, root.parent.id, is_left)
        else:
            old_tree.set_root_leaf(new_tree.root.cls)
    else:
        q = [(root, new_tree.root)]
        root.feature = new_tree.root.feature
        root.threshold = new_tree.root.threshold
        root.is_categorical = new_tree.root.is_categorical

        while q:
            o_r, n_r = q.pop()

            for c_p, c_c in [(True, n_r.left), (False, n_r.right)]:
                if c_c.is_leaf:
                    if c_c.cls.startswith("-"):
                        old_tree.add_leaf(c_c.cls[1:], o_r.id, c_p)
                    else:
                        leaf_id = int(c_c.cls)
                        if leaf_id not in used_leaves:
                            used_leaves.add(leaf_id)
                            if c_p:
                                o_r.left = old_tree.nodes[int(c_c.cls)]
                                old_tree.nodes[int(c_c.cls)].parent = o_r
                            else:
                                o_r.right = old_tree.nodes[int(c_c.cls)]
                                old_tree.nodes[int(c_c.cls)].parent = o_r
                        else:
                            duplicate(o_r, leaf_id, c_p)
                            duplicated = True
                else:
                    n_r = old_tree.add_node(c_c.feature, c_c.threshold, o_r.id, c_p, c_c.is_categorical)
                    q.append((n_r, c_c))

    if instance is not None and duplicated:
        # TODO: This is necessary if sub-trees are duplicated, but only has to be performed for the sub-tree
        old_tree.clean(instance)


def _get_max_bound(size, sample_limit):
    new_ub = -1
    for cb, sl in enumerate(sample_limit):
        if sl < size:
            new_ub = cb
    return new_ub


def leaf_select(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, encoding, slv, opt_size=False, opt_slim=False, multiclass=False):
    last_idx = path_idx
    while path_idx < len(path):
        c_d = path[path_idx].get_depth()
        if c_d > depth_limit or len(assigned[path[path_idx].id]) > sample_limit[c_d]:
            break
        last_idx = path_idx
        path_idx += 1

    node = path[last_idx]
    c_d = node.get_depth()
    if c_d <= 2:
        return False, last_idx

    new_ub = _get_max_bound(len(assigned[node.id]), sample_limit)

    if new_ub < 1:
        return False, last_idx

    new_instance = ClassificationInstance()
    for s in assigned[node.id]:
        n_s = s.copy(new_instance)
        n_s.cls = f"-{n_s.cls}"
        new_instance.add_example(n_s)

    new_instance.finish()
    if len(new_instance.examples) == 0 or encoding.estimate_size(new_instance, c_d-1) > literal_limit:
        return False, last_idx

    if encoding.is_sat():
        new_tree = bs.run(encoding, new_instance, slv, start_bound=min(new_ub, c_d - 1), timeout=time_limit, ub=min(new_ub, c_d - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)
    else:
        new_tree = encoding.run(new_instance, start_bound=min(new_ub, c_d - 1), timeout=time_limit,
               ub=min(new_ub, c_d - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)

    if new_tree is None:
        return False, last_idx
    else:
        stitch(tree, new_tree, node, None)
        return True, last_idx


def leaf_rearrange(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, encoding, slv, opt_size=False, opt_slim=False, multiclass=False):
    prev_instance = None
    prev_idx = path_idx

    while path_idx < len(path):
        c_d = path[path_idx].get_depth()
        if c_d > depth_limit:
            break

        new_instance = build_unique_set(path[path_idx], assigned[path[path_idx].id], instance.examples)
        if new_instance is None:
            break

        if encoding.estimate_size(new_instance[0], c_d) > literal_limit:
            break
        if len(new_instance[0].examples) > sample_limit[c_d]:
            if prev_instance is not None or len(new_instance[0].examples) > sample_limit[1]:
                break

        prev_instance = new_instance
        prev_idx = path_idx
        path_idx += 1

    if prev_instance is not None and len(prev_instance[0].examples) > 0:
        node = path[prev_idx]
        new_instance, feature_map, c_leafs, _ = prev_instance
        cd = node.get_depth()
        if cd < 2:
            return False, prev_idx
        new_ub = _get_max_bound(len(prev_instance[0].examples), sample_limit)
        if new_ub < 1:
            return False, prev_idx

        # Solve instance
        if encoding.is_sat():
            new_tree = bs.run(encoding, new_instance, slv, start_bound=min(new_ub, cd-1), timeout=time_limit, ub=min(new_ub, cd-1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)
        else:
            new_tree = encoding.run(new_instance, start_bound=min(new_ub, cd - 1), timeout=time_limit,
                              ub=min(new_ub, cd - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)

        # Either the branch is done, or
        if new_tree is None:
            return False, prev_idx
        else:
            new_instance.unreduce(tree)
            # Correct features
            q = [new_tree.root]
            while q:
                c_q = q.pop()
                if not c_q.is_leaf:
                    c_q.feature = feature_map[c_q.feature - 1]
                    q.append(c_q.left)
                    q.append(c_q.right)

            # Clean tree
            stitch(tree, new_tree, node, None)

            return True, prev_idx

    return False, prev_idx


def reduced_leaf(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, time_limit, encoding, slv, opt_size=False, opt_slim=False, multiclass=False):
    prev_instance = None
    prev_idx = path_idx

    while True:
        if path_idx >= len(path):
            break
        c_d = path[path_idx].get_depth()
        if c_d > depth_limit:
            break

        new_instance = ClassificationInstance()
        for s in assigned[path[path_idx].id]:
            n_s = s.copy(new_instance)
            n_s.cls = f"-{n_s.cls}"
            new_instance.add_example(n_s)
        new_instance.finish()
        if len(new_instance.examples) == 0:
            break

        new_instance.reduce_with_key()

        if encoding.estimate_size(new_instance, c_d-1) > literal_limit:
            break
        if len(new_instance.examples) > sample_limit[c_d]:
            if prev_instance is not None or len(new_instance.examples) > sample_limit[1]:
                break

        prev_instance = new_instance
        prev_idx = path_idx
        path_idx += 1

    if prev_instance is not None:
        node = path[prev_idx]
        nd = node.get_depth()
        new_ub = _get_max_bound(len(prev_instance.examples), sample_limit)
        if new_ub < 1:
            return False, prev_idx

        if encoding.is_sat():
            new_tree = bs.run(encoding, prev_instance, slv, start_bound=min(new_ub, nd - 1), timeout=time_limit,
                                  ub=min(new_ub, nd - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)
        else:
            new_tree = encoding.run(prev_instance, start_bound=min(new_ub, nd - 1), timeout=time_limit,
                              ub=min(new_ub, nd - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)

        if new_tree is not None:
            prev_instance.unreduce(new_tree)
            stitch(tree, new_tree, node, None)

            return True, prev_idx

    return False, prev_idx


def mid_reduced(tree, instance, path_idx, path, assigned, depth_limit, sample_limit, reduce, time_limit, encoding, slv, opt_size=False, opt_slim=False, multiclass=False):
    # Exclude nodes with fewer than limit samples, as this will be handled by the leaf methods
    if path[path_idx].is_leaf:
        return False, path_idx

    c_parent = path[path_idx]
    new_instance, i_depth = build_reduced_set(c_parent, tree, instance.examples, assigned, depth_limit, sample_limit, reduce, encoding)

    if new_instance is None or len(new_instance.examples) == 0:
        return False, path_idx
    new_ub = _get_max_bound(len(new_instance.examples), sample_limit)
    if new_ub < 1:
        return False, path_idx

    if encoding.is_sat():
        new_tree = bs.run(encoding, new_instance, slv, start_bound=min(new_ub, i_depth - 1), timeout=time_limit,
                          ub=min(new_ub, i_depth - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)
    else:
        new_tree = encoding.run(new_instance, start_bound=min(new_ub, i_depth - 1), timeout=time_limit,
                          ub=min(new_ub, i_depth - 1), opt_size=opt_size, slim=opt_slim, multiclass=multiclass)

    if new_tree is not None:
        new_instance.unreduce(new_tree)
        # Stitch the new tree in the middle
        stitch(tree, new_tree, c_parent, instance)
        return True, path_idx

    return False, path_idx
