import sys
import time
from sys import maxsize

import nonbinary.nonbinary_instance
from nonbinary.nonbinary_instance import ClassificationInstance


def build_unique_set(parameters, root, samples, reduce, limit=maxsize, reduce_limit=5000):
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
            c_features.add((c_q.feature, c_q.threshold, c_q.is_categorical))

    def create_local_instance():
        new_local_instance = ClassificationInstance()
        new_local_instance.is_categorical.update(parameters.instance.is_categorical)
        for s in samples:
            if root.decide(s)[0] != s.cls:  # Skip misclassified
                continue
            new_s = s.copy(new_local_instance)
            new_s.cls = f"-{s.cls}"
            new_local_instance.add_example(new_s)
        new_local_instance.finish()
        return new_local_instance

    new_instance = create_local_instance()
    if reduce and len(new_instance.examples) < reduce_limit:
        #print(f"{len(new_instance.examples)}")
        new_instance.reduce_with_key(numeric_full=parameters.reduce_numeric_full or parameters.use_smt,
                                     cat_full=parameters.reduce_categoric_full or parameters.use_smt,
                                     reduce_alternate=parameters.reduce_alternate)
        #print(f"{len(new_instance.examples)}")
    else:
        if not (parameters.use_smt or parameters.reduce_numeric_full or parameters.reduce_categoric_full):
            feature_key = c_features
        else:
            feature_key = set()
            for c_f, c_v, c_c in c_features:
                if c_f in new_instance.is_categorical:
                    if parameters.use_smt or parameters.reduce_categoric_full:
                        feature_key.add((c_f, None, None))
                    else:
                        feature_key.add((c_f, c_v, c_c))
                else:
                    if parameters.use_smt or parameters.reduce_numeric_full:
                        feature_key.add((c_f, None, None))
                    else:
                        feature_key.add((c_f, c_v, c_c))

        new_instance.reduce(feature_key)

    if reduce and len(new_instance.examples) > reduce_limit and parameters.decide_instance(new_instance, depth):
        new_instance = create_local_instance()
        new_instance.reduce_with_key(numeric_full=parameters.reduce_numeric_full or parameters.use_smt,
                                     cat_full=parameters.reduce_categoric_full or parameters.use_smt,
                                     reduce_alternate=parameters.reduce_alternate)
    elif reduce and len(new_instance.examples) > reduce_limit:
        return None, 0, 0

    return new_instance, len(c_leafs), depth


def build_reduced_set(parameters, root, assigned, reduce, limit=sys.maxsize, reduce_limit=5000, duplicate=False):
    max_dist = root.get_depth()
    q = [[] for _ in range(0, max_dist+1)]
    q[max_dist].append((0, root))

    features = set()
    last_instance = None
    cnt = 0
    cnt_internal = 0
    frontier = {root.id}
    max_depth = 0
    c_max_depth = 0

    reduced = False
    duplicate_instance = None

    def create_local_instance(cm):
        new_local_instance = ClassificationInstance()
        new_local_instance.is_categorical.update(parameters.instance.is_categorical)

        for cs in assigned[root.id]:
            if root.decide(cs)[0] != cs.cls:  # Skip misclassified
                continue
            n_s = cs.copy(new_local_instance)
            n_s.cls, is_leaf = cm[cs.id]
            if not is_leaf and cs.cls in classes:
                n_s.surrogate_cls = f"-{cs.cls}"
            new_local_instance.add_example(n_s)
        new_local_instance.class_sizes = class_sizes
        new_local_instance.is_categorical.update(parameters.instance.is_categorical)
        new_local_instance.finish()
        return new_local_instance

    class_mapping = None
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
                features.add((new_root.feature, new_root.threshold, new_root.is_categorical))
                q[new_root.left.get_depth()].append((c_depth + 1, new_root.left))
                q[new_root.right.get_depth()].append((c_depth + 1, new_root.right))

                frontier.remove(new_root.id)
                frontier.add(new_root.left.id)
                frontier.add(new_root.right.id)

        # Complete with leafs, if possible, as they won't add any samples
        for cl in list(frontier):
            c_n = parameters.tree.nodes[cl]
            if not c_n.is_leaf and c_n.left.is_leaf and c_n.right.is_leaf:
                frontier.remove(cl)
                features.add((c_n.feature, c_n.threshold, c_n.is_categorical))

                frontier.add(c_n.left.id)
                frontier.add(c_n.right.id)

        if c_max_depth > parameters.maximum_depth or c_max_depth > limit:
            break

        if cnt >= 3:
            class_sizes = {}
            class_mapping = {}
            cnt_internal = 0
            classes = set()
            for c_leaf in frontier:
                for s in assigned[c_leaf]:
                    if parameters.tree.nodes[c_leaf].is_leaf:
                        class_mapping[s.id] = (f"-{parameters.tree.nodes[c_leaf].cls}", True)
                        classes.add(parameters.tree.nodes[c_leaf].cls)
                    else:
                        cnt_internal += 1
                        class_mapping[s.id] = (f"{c_leaf}", False)
                        class_sizes[f"{c_leaf}"] = parameters.tree.nodes[c_leaf].get_leaves()


            # If all "leaves" are leaves, this method is not required, as it will be handled by separate improvements
            if cnt_internal > 0:
                new_instance = create_local_instance(class_mapping)
                if duplicate:
                    duplicate_instance = new_instance.copy()

                if reduce and len(new_instance.examples) < reduce_limit:
                    #print(f"{len(new_instance.examples)}")
                    new_instance.reduce_with_key(numeric_full=parameters.reduce_numeric_full or parameters.use_smt,
                                                 cat_full=parameters.reduce_categoric_full or parameters.use_smt,
                                                 reduce_alternate=parameters.reduce_alternate)
                    #print(f"{len(new_instance.examples)}")
                else:
                    if not (parameters.use_smt or parameters.reduce_numeric_full or parameters.reduce_categoric_full):
                        feature_key = set(features)
                    else:
                        feature_key = set()
                        for c_f, c_v, c_c in features:
                            if c_f in new_instance.is_categorical:
                                if parameters.use_smt or parameters.reduce_categoric_full:
                                    feature_key.add((c_f, None, None))
                                else:
                                    feature_key.add((c_f, c_v, c_c))
                            else:
                                if parameters.use_smt or parameters.reduce_numeric_full:
                                    feature_key.add((c_f, None, None))
                                else:
                                    feature_key.add((c_f, c_v, c_c))

                    new_instance.reduce(feature_key)

                if parameters.decide_instance(new_instance, c_max_depth) or \
                        (last_instance is None and parameters.decide_instance(new_instance, 1)):
                    last_instance = new_instance
                    if reduce and len(new_instance.examples) < reduce_limit:
                        reduced = True
                    else:
                        reduced = False
                    max_depth = c_max_depth
                else:
                    q = None

    if reduce and not reduced and cnt >= 3 and cnt_internal > 0:
        new_instance = create_local_instance(class_mapping)
        if duplicate:
            duplicate_instance = new_instance.copy()

        if reduce and len(new_instance.examples) < reduce_limit:
            # print(f"{len(new_instance.examples)}")
            new_instance.reduce_with_key(numeric_full=parameters.reduce_numeric_full or parameters.use_smt,
                                         cat_full=parameters.reduce_categoric_full or parameters.use_smt,
                                         reduce_alternate=parameters.reduce_alternate)
            if parameters.decide_instance(new_instance, c_max_depth) or \
                    (last_instance is None and parameters.decide_instance(new_instance, 1)):
                return new_instance, max_depth, len(frontier)
            else:
                return None, 0, 1

    return last_instance, max_depth, len(frontier), duplicate_instance


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


def leaf_select(parameters, node, assigned, instance):
    if len(assigned[node.id]) > parameters.maximum_examples:
        return None, False
    c_d = node.get_depth()
    if c_d < 2:
        return False, True

    new_instance = ClassificationInstance()
    for s in assigned[node.id]:
        n_s = s.copy(new_instance)
        for f, v in enumerate(n_s.features):
            if v == "?":
                n_s.features[f] = instance.domains_max[f]

        n_s.cls = f"-{n_s.cls}"
        new_instance.add_example(n_s)
    new_instance.is_categorical.update(parameters.instance.is_categorical)
    new_instance.finish()

    new_ub = parameters.get_max_bound(new_instance)
    original_ub = new_ub >= c_d
    if new_ub < 1:
        return None, original_ub

    leaves = node.get_leaves()
    new_tree, is_sat = parameters.call_solver(new_instance, new_ub, c_d, leaves)

    if is_sat is None:
        # Wait after a memory out, to ensure resources are cleaned up
        time.sleep(30)

    if new_tree is None:
        return is_sat, original_ub
    else:
        stitch(parameters.tree, new_tree, node, None)
        return True, original_ub


def leaf_reduced(parameters, node, assigned, instance, reduce=False, reduce_limit=10000):
    new_instance, leaves, cd = build_unique_set(parameters, node, assigned[node.id], reduce=reduce, reduce_limit=reduce_limit)
    if new_instance is None:
        return None, True, False

    if cd < 2:
        return False, True, False

    new_ub = parameters.get_max_bound(new_instance)
    original_ub = new_ub >= cd
    if new_ub < 1:
        return False, True, False

    for cs in new_instance.examples:
        for f, v in enumerate(cs.features):
            if v == "?":
                cs.features[f] = instance.domains_max[f]

    # Solve instance
    new_tree, is_sat = parameters.call_solver(new_instance, new_ub, cd, leaves)

    # Either the branch is done, or
    if new_tree is not None:
        new_instance.unreduce(new_tree)
        stitch(parameters.tree, new_tree, node, None)
        return True, original_ub, True
    elif is_sat is None:
        time.sleep(30)

    return is_sat, original_ub, True


def mid_reduced(parameters, node, assigned, instance, reduce, reduce_limit=10000, rerun=False):
    # Exclude nodes with fewer than limit samples, as this will be handled by the leaf methods
    if node.is_leaf:
        return False

    c_parent = node
    limits = sys.maxsize
    new_tree = None
    new_instance = None

    while limits >= 2 and new_tree is None:
        new_instance, i_depth, leaves, duplicate_instance = \
            build_reduced_set(parameters, c_parent, assigned, reduce, limits, reduce_limit, duplicate=rerun)

        if new_instance is None or len(new_instance.examples) == 0:
            return None, True

        new_ub = parameters.get_max_bound(new_instance)
        original_ub = new_ub >= i_depth
        if new_ub < 1:
            return None, original_ub

        for cs in new_instance.examples:
            for f, v in enumerate(cs.features):
                if v == "?":
                    cs.features[f] = instance.domains_max[f]

        new_tree, is_sat = parameters.call_solver(new_instance, new_ub, i_depth, leaves)

        if new_tree is not None:
            new_instance.unreduce(new_tree)

            # Stitch the new tree in the middle
            stitch(parameters.tree, new_tree, c_parent, parameters.instance)
            return True, original_ub
        elif is_sat is None:
            time.sleep(30)

        if rerun and len(new_instance.examples) and len(new_instance.examples) <= reduce_limit:
            duplicate_instance.reduce_with_key(numeric_full=parameters.reduce_numeric_full or parameters.use_smt,
                                         cat_full=parameters.reduce_categoric_full or parameters.use_smt,
                                         reduce_alternate=parameters.reduce_alternate)
            new_tree, is_sat = parameters.call_solver(duplicate_instance, new_ub, i_depth, leaves)
            if new_tree is not None:
                new_instance.unreduce(new_tree)

                # Stitch the new tree in the middle
                stitch(parameters.tree, new_tree, c_parent, parameters.instance)
                return True, original_ub
            elif is_sat is None:
                time.sleep(30)

        return is_sat, original_ub

    return None, True
