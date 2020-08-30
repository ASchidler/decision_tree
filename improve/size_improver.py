import improve.improver as improver
from bdd_instance import BddInstance
import bdd_instance
import sat_tools
import tree_node_encoding


def build_runner(tmp_dir):
    return sat_tools.SatRunner(tree_node_encoding.TreeEncoding, sat_tools.GlucoseSolver(), base_path=tmp_dir)


def find_tree(root):
    q = [root]
    nodes = []
    while q:
        c_n = q.pop()
        nodes.append(c_n)
        if not c_n.is_leaf:
            q.append(c_n.left)
            q.append(c_n.right)
    return nodes


def find_features(root):
    features = []
    q = [root]
    while q:
        c_n = q.pop()
        if not c_n.is_leaf:
            features.append(c_n.feature)
            q.append(c_n.left)
            q.append(c_n.right)
    return features


def leaf_select(tree, instance, path_idx, path, assigned, size_limit, sample_limit, tmp_dir="."):
    runner = build_runner(tmp_dir)

    last_idx = path_idx
    last_size = 0
    while last_idx < len(path) - 1:
        if len(assigned[path[last_idx + 1].id]) <= sample_limit:
            c_size = len(find_tree(path[last_idx+1]))
            if c_size <= size_limit:
                last_idx += 1
                last_size = c_size
            else:
                break
        else:
            break

    if last_idx <= path_idx:
        return False, path_idx

    new_instance = BddInstance()
    node = path[last_idx]
    for s in assigned[node.id]:
        new_instance.add_example(instance.examples[s].copy())
    print(f"{last_size} {len(new_instance.examples)}")
    if len(new_instance.examples) == 0:
        return False, last_idx

    new_tree, _ = runner.run(new_instance, last_size-2, u_bound=last_size-2)
    if new_tree is None:
        return False, last_idx
    else:
        improver.replace(tree, new_tree, node)
        return True, last_idx


def leaf_reduce(tree, instance, path_idx, path, assigned, size_limit, sample_limit, reduce, tmp_dir="."):
    runner = build_runner(tmp_dir)

    last_idx = path_idx
    last_size = 0
    last_instance = None

    while last_idx < len(path) - 1:
        c_size = len(find_tree(path[last_idx + 1]))

        if c_size <= size_limit:
            new_instance = BddInstance()
            node = path[last_idx + 1]
            for s in assigned[node.id]:
                new_instance.add_example(instance.examples[s].copy())

            if reduce:
                bdd_instance.reduce(new_instance, randomized_runs=1)
            else:
                bdd_instance.reduce(new_instance, min_key=set(find_features(path[last_idx + 1])))

            if len(new_instance.examples) <= sample_limit:
                last_idx += 1
                last_size = c_size
                last_instance = new_instance
            else:
                break
        else:
            break

    if last_idx <= path_idx:
        return False, max(0, path_idx-1)

    print(f"{last_size} {len(last_instance.examples)}")
    new_tree, _ = runner.run(last_instance, last_size-2, u_bound=last_size-2)
    if new_tree is None:
        return False, max(0, path_idx-1)
    else:
        node = path[last_idx]
        last_instance.unreduce_instance(new_tree)
        improver.replace(tree, new_tree, node)
        return True, last_idx


def mid_reduce(tree, instance, path_idx, path, assigned, size_limit, sample_limit, reduce, tmp_dir="."):
    root = path[path_idx]

    if root.is_leaf:
        return False, path_idx

    new_instance = None
    frontier = {root.left.id, root.right.id}
    q = [root.left, root.right]
    features = [root.feature]
    if not root.left.is_leaf:
        features.append(root.left.feature)
    if not root.right.is_leaf:
        features.append(root.right.feature)

    cnt = 3

    while q and cnt < size_limit:
        c_n = q.pop()
        if c_n.is_leaf:
            continue

        c_instance = BddInstance()
        class_mapper = {}
        tmp_features = []

        for c_id in frontier:
            if tree.nodes[c_id].is_leaf:
                for e in assigned[c_id]:
                    class_mapper[e] = -2 if tree.nodes[c_id].cls else -1
            elif c_id != c_n.id:
                for e in assigned[c_id]:
                    class_mapper[e] = c_id

        for n_n in [c_n.left, c_n.right]:
            if n_n.is_leaf:
                for e in assigned[n_n.id]:
                    class_mapper[e] = -2 if n_n.cls else -1
            else:
                tmp_features.append(n_n.feature)
                for e in assigned[n_n.id]:
                    class_mapper[e] = n_n.id

        for e in assigned[root.id]:
            c_instance.add_example(bdd_instance.BddExamples(instance.examples[e].features, class_mapper[e], instance.examples[e].id))

        if reduce:
            bdd_instance.reduce(c_instance, randomized_runs=1)
        else:
            bdd_instance.reduce(c_instance, min_key={*features, *tmp_features})

        if len(c_instance.examples) <= sample_limit:
            new_instance = c_instance
            cnt += 2
            frontier.remove(c_n.id)
            frontier.add(c_n.left.id)
            frontier.add(c_n.right.id)
            q.append(c_n.left)
            q.append(c_n.right)
            features.extend(tmp_features)

    if new_instance is not None:
        print(f"{root.id} {reduce} {cnt} {len(new_instance.examples)}")
        runner = build_runner(tmp_dir)
        new_tree, _ = runner.run(new_instance, cnt - 2, u_bound=cnt - 2)
        if new_tree is None:
            return False, path_idx
        else:
            new_instance.unreduce_instance(new_tree)
            print(f"{new_tree.get_accuracy(new_instance.examples)}")
            improver.stitch(tree, new_tree, root)
            return True, path_idx
    return False, path_idx