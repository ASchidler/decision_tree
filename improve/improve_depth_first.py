import improve.improver as improver
import time

sample_limit = [400, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200,
                200, 150, 100, 100, 100, 0]
depth_limit = 15
reduce_runs = 1


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

        if c_d > c_max[0] and c_n.id not in ignore:
            c_max = (c_d, c_n)

        if not c_n.is_leaf:
            if c_n.left.id:
                parent[c_n.left.id] = c_n
                q.append((c_d+1, c_n.left))
            if c_n.right.id:
                parent[c_n.right.id] = c_n
                q.append((c_d+1, c_n.right))

    if c_max[1] is None:
        return -1, None, None

    c_node = c_max[1]
    path = []

    while c_node is not None:
        path.append(c_node)
        c_node = parent[c_node.id]

    return c_max[0], c_max[1], path


def run(tree, instance, test, tmp_dir="."):
    # Select nodes based on the depth
    c_ignore = set()

    start_time = time.time()

    def process_change(ignore, c_path, mth):
        print(f"Time: {time.time() - start_time:.4f}\t"
              f"Training {tree.get_accuracy(instance.examples):.4f}\t"
              f"Test {tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {tree.get_depth():03}\t"
              f"Avg {tree.get_avg_depth():03.4f}\t"
              f"Nodes {tree.get_nodes()}\t"
              f"Method {mth}")

        for c_n in c_path:
            ignore.discard(c_n.id)

    while True:
        new_max_d, new_max_n, new_max_p = find_deepest_leaf(tree, c_ignore)

        # No nodes left
        if new_max_n is None:
            break

        assigned = assign_samples(tree, instance)
        done = False

        # First try to find root
        result, select_idx = improver.leaf_select(tree, instance, 0, new_max_p, assigned, depth_limit, sample_limit, tmp_dir=tmp_dir)
        if result:
            process_change(c_ignore, new_max_p, "ls")
            continue

        max_leaf_idx = 0
        result, idx = improver.leaf_rearrange(tree, instance, select_idx, new_max_p, assigned, depth_limit, sample_limit, tmp_dir=tmp_dir)
        max_leaf_idx = max(max_leaf_idx, idx)
        if result:
            process_change(c_ignore, new_max_p, "la")
            continue

        for _ in range(0, reduce_runs):
            result, idx = improver.reduced_leaf(tree, instance, select_idx, new_max_p, assigned, depth_limit, sample_limit, tmp_dir=tmp_dir)
            max_leaf_idx = max(max_leaf_idx, idx)
            if result:
                process_change(c_ignore, new_max_p, "lr")
                done = True
                break
        if done:
            continue

        # Try reducing starting from the root
        for i in range(len(new_max_p)-1, max_leaf_idx, -1):
            if new_max_p[i].id in c_ignore:
                continue

            result, idx = improver.mid_rearrange(tree, instance, i, new_max_p, assigned, depth_limit, sample_limit, tmp_dir=tmp_dir)
            if result:
                process_change(c_ignore, new_max_p, "ma")
                done = True
                break
            # result, idx = improver.mid_reduced(tree, instance, i, new_max_p, assigned, False, depth_limit, sample_limit)
            # if result:
            #     process_change(c_ignore, new_max_p)
            #     done = True
            #     break

            for _ in range(0, reduce_runs):
                result, idx = improver.mid_reduced(tree, instance, i, new_max_p, assigned, True,
                                                   sample_limit, depth_limit, tmp_dir=tmp_dir)
                if result:
                    process_change(c_ignore, new_max_p, "mr")
                    done = True
                    break
            if done:
                break
        if done:
            continue

        # Could not improve, ignore tried nodes for future tries
        #print(f"None {new_max_n.id}")
        for c_n in new_max_p:
            c_ignore.add(c_n.id)
