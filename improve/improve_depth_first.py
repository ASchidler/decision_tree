import improve.improver as improver
import time

sample_limit = 50
depth_limit = 12
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


def run(tree, instance, test):
    # Select nodes based on the depth
    c_ignore = set()

    start_time = time.time()

    def process_change(ignore, c_path):
        print(f"Time: {time.time() - start_time:.4f}\t"
              f"Training {tree.get_accuracy(instance.examples):.4f}\t"
              f"Test {tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {tree.get_depth()}\t"
              f"Avg {tree.get_avg_depth():.4f}\t"
              f"Nodes {tree.get_nodes()}")

        for c_n in c_path:
            ignore.discard(c_n.id)

    while True:
        new_max_d, new_max_n, new_max_p = find_deepest_leaf(tree, c_ignore)

        # No nodes left
        if new_max_n is None:
            break

        assigned = assign_samples(tree, instance)
        done_idx = []

        # First try to find root
        result, select_idx = improver.leaf_select(tree, instance, 0, new_max_p, assigned, depth_limit, sample_limit)
        done_idx.append(select_idx)
        if result:
            process_change(c_ignore, new_max_p)
            continue

        result, idx = improver.leaf_rearrange(tree, instance, select_idx, new_max_p, assigned, depth_limit, sample_limit)
        done_idx.append(idx)
        if result:
            process_change(c_ignore, new_max_p)
            continue

        for _ in range(0, reduce_runs):
            result, idx = improver.reduced_leaf(tree, instance, select_idx, new_max_p, assigned, depth_limit, sample_limit)
            done_idx.append(idx)
            if result:
                process_change(c_ignore, new_max_p)
                continue

        # Could not improve, ignore tried nodes for future tries
        #print(f"None {new_max_n.id}")
        done_idx.append(0)
        for c_idx in done_idx:
            if c_idx >= 0:
                for i in range(0, c_idx + 1):
                    c_ignore.add(new_max_p[i].id)
