import time
import improve.improver as improver


sample_limit = 100
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


def find_leaves(tree, done):
    q = [(0, tree.root)]
    c_max = (-1, None)
    parent = {tree.root.id: None}

    while q:
        c_d, c_n = q.pop()

        if c_n.is_leaf:
            if c_d > c_max[0] and c_n.id not in done:
                c_max = (c_d, c_n)

        if not c_n.is_leaf:
            if c_n.left.id:
                parent[c_n.left.id] = c_n
                q.append((c_d + 1, c_n.left))
            if c_n.right.id:
                parent[c_n.right.id] = c_n
                q.append((c_d + 1, c_n.right))

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

    start_time = time.time()

    def process_change():
        print(f"Time: {time.time() - start_time:.4f}\t"
              f"Training {tree.get_accuracy(instance.examples):.4f}\t"
              f"Test {tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {tree.get_depth():03}\t"
              f"Avg {tree.get_avg_depth():03.4f}\t"
              f"Nodes {tree.get_nodes()}")

    def add_done(node, c_done):
        q = [node]

        while q:
            c_node = q.pop()
            c_done.add(c_node.id)

            if not c_node.is_leaf:
                q.append(c_node.left)
                q.append(c_node.right)

    changed = True
    while changed:
        changed = False
        # First do leaf selects (quick wins)

        def process_leaves(mode):
            done = set()
            assigned = assign_samples(tree, instance)
            any_change = False

            while True:
                new_max_d, new_max_n, new_max_p = find_leaves(tree, done)

                # No nodes left
                if new_max_n is None:
                    break

                # First try to find root
                if mode == 0:
                    result, idx = improver.leaf_select(tree, instance, 0, new_max_p, assigned, depth_limit, sample_limit)
                elif mode == 1:
                    result, idx = improver.leaf_rearrange(tree, instance, 0, new_max_p, assigned, depth_limit,
                                                          sample_limit)
                else:
                    result, idx = improver.reduced_leaf(tree, instance, 0, new_max_p, assigned, depth_limit, sample_limit)

                add_done(new_max_p[idx], done)
                if result:
                    any_change = True
                    process_change()
                    assigned = assign_samples(tree, instance)
            return any_change

        changed1 = process_leaves(0)
        changed2 = process_leaves(1)
        changed3 = process_leaves(2)
        changed = changed1 or changed2 or changed3

        # # Try reducing starting from the root
        # for i in range(len(new_max_p)-1, max_leaf_idx, -1):
        #     if new_max_p[i].id in c_ignore:
        #         continue
        #
        #     result, idx = improver.mid_rearrange(tree, instance, i, new_max_p, assigned, depth_limit, sample_limit)
        #     if result:
        #         process_change(c_ignore, new_max_p)
        #         done = True
        #         break
        #     # result, idx = improver.mid_reduced(tree, instance, i, new_max_p, assigned, False, depth_limit, sample_limit)
        #     # if result:
        #     #     process_change(c_ignore, new_max_p)
        #     #     done = True
        #     #     break
        #
        #     for _ in range(0, reduce_runs):
        #         result, idx = improver.mid_reduced(tree, instance, i, new_max_p, assigned, True, depth_limit,
        #                                            sample_limit)
        #         if result:
        #             process_change(c_ignore, new_max_p)
        #             done = True
        #             break
        #     if done:
        #         break
        # if done:
        #     continue
        #
        # # Could not improve, ignore tried nodes for future tries
        # #print(f"None {new_max_n.id}")
        # for c_n in new_max_p:
        #     c_ignore.add(c_n.id)
