import improve.improver as improver
import time
import sys
import decision_tree
import time

sample_limit_short = [175,
                      175, 175, 175, 175, 175,
                      175, 175, 175, 150, 120,
                      70, 70, 70, 70, 70,
                      0]

sample_limit_mid = [300,
                    300, 300, 300, 270, 270,
                    270, 270, 270, 270, 250,
                    200, 200, 175, 90, 90,
                    0]

sample_limit_long = [500,
                     500, 500, 500, 500, 500,
                     500, 500, 500, 500, 500,
                     400, 340, 250, 250, 215,
                     105, 105, 105, 105, 105,
                     105, 105, 105, 105, 105,
                     105, 105, 105, 105, 105,
                     105, 105, 105, 105, 105,
                     105, 105, 105, 105, 105]

time_limits = [60, 300, 800]
depth_limits = [12, 15, 29]

reduce_runs = 1


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
        return None

    c_node = c_max[1]
    path = []

    while c_node is not None:
        path.append(c_node)
        c_node = parent[c_node.id]

    return path


def clear_ignore(ignore, root):
    q = [root]

    while q:
        c_n = q.pop()
        ignore.discard(c_n.id)
        if not c_n.is_leaf:
            q.append(c_n.left)
            q.append(c_n.right)


def run(tree, instance, test, limit_idx=1, pt=False, timelimit=0, opt_size=False):
    sample_limit = [sample_limit_short, sample_limit_mid, sample_limit_long][limit_idx]
    time_limit = time_limits[limit_idx]
    depth_limit = depth_limits[limit_idx]

    # Select nodes based on the depth
    c_ignore = set()
    c_ignore_reduce = set()

    start_time = time.time()

    def process_change(mth):
        print(f"Time: {time.time() - start_time:.4f}\t"
              f"Training {tree.get_accuracy(instance.examples):.4f}\t"
              f"Test {tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {tree.get_depth():03}\t"
              f"Avg {tree.get_avg_depth():03.4f}\t"
              f"Nodes {tree.get_nodes()}\t"
              f"Method {mth}")
        sys.stdout.flush()

    assigned = tree.assign_samples(instance)
    tree_size = tree.get_nodes()
    while tree_size > len(c_ignore_reduce):
        allow_reduction = False
        pth = find_deepest_leaf(tree, c_ignore)

        if pth is None:
            allow_reduction = True
            pth = find_deepest_leaf(tree, c_ignore_reduce)
            if pth is None:
                return

        while pth:
            root = pth.pop()

            if (not allow_reduction and root.id in c_ignore) or (allow_reduction and root.id in c_ignore_reduce):
                continue

            op = None
            if not allow_reduction:
                # result, _ = improver.mid_rearrange(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit)
                result, _ = improver.leaf_select(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit, opt_size=opt_size)
                if result:
                    op = "ls"
                if not result:
                    result, _ = improver.leaf_rearrange(tree, instance, 0, [root], assigned, depth_limit, sample_limit,
                                                        time_limit, opt_size=opt_size)
                    if result:
                        op = "la"

                if not result:
                    result, _ = improver.mid_reduced(tree, instance, 0, [root], assigned, False, sample_limit,
                                                     depth_limit, time_limit, opt_size=opt_size)
                    if result:
                        op = "ma"
            else:
                result, _ = improver.reduced_leaf(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit, opt_size=opt_size)
                if result:
                    op = "lr"
                if not result:
                    result, _ = improver.mid_reduced(tree, instance, 0, [root], assigned, True, sample_limit,
                                                     depth_limit, time_limit, opt_size=opt_size)
                    if result:
                        op = "mr"

            if 0 < timelimit < (time.time() - start_time):
                return

            if not allow_reduction:
                c_ignore.add(root.id)
            else:
                c_ignore_reduce.add(root.id)
            if result:
                process_change(op)
                # TODO: This could be more efficient... We only have to re-compute assigned from the current root!
                assigned = tree.assign_samples(instance)
                tree_size = tree.get_nodes()
                # May have been replaced if we reduce the tree to a leaf
                if tree.nodes[root.id] == root:
                    clear_ignore(c_ignore, root)
                    clear_ignore(c_ignore_reduce, root)
                # Break as the path may have become invalid
                break
            else:
                print("None")

        # def find_mid(root, assigned):
        #     if root.id in c_ignore:
        #         return assigned
        #
        #     op = None
        #     # result, _ = improver.mid_rearrange(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit)
        #     result, _ = improver.leaf_select(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit)
        #     if result:
        #         op = "ls"
        #     if not result:
        #         result, _ = improver.leaf_rearrange(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit)
        #         if result:
        #             op = "la"
        #
        #     if not result:
        #         result, _ = improver.mid_reduced(tree, instance, 0, [root], assigned, False, sample_limit,
        #                                          depth_limit, time_limit)
        #         if result:
        #             op = "ma"
        #
        #     if result:
        #         process_change(op)
        #         # TODO: This could be more efficient...
        #         assigned = tree.assign_samples(instance)
        #     else:
        #         print("None")
        #
        #     c_ignore.add(root.id)
        #     if not root.is_leaf:
        #         assigned = find_mid(root.left, assigned)
        #         assigned = find_mid(root.right, assigned)
        #
        #     return assigned
        #
        # find_mid(tree.root, tree.assign_samples(instance))
        #
        # for op in ["ls", "la", "lr"]:
        #     c_branch = []
        #
        #     def find_leaf(root, assigned):
        #         if 0 < timelimit < (time.time() - start_time):
        #             return
        #
        #         c_branch.append(root)
        #         if not root.is_leaf:
        #             r1, assigned = find_leaf(root.left, assigned)
        #             # root.right may become None if we reduced the sub-tree to a leaf.
        #             if r1 >= len(c_branch) - 1 and root.right is not None:
        #                 r1, assigned = find_leaf(root.right, assigned)
        #             c_branch.pop()
        #             return r1, assigned
        #         else:
        #             if op == "ls":
        #                 result, select_idx = improver.leaf_select(tree, instance, 0, list(reversed(c_branch)), assigned, depth_limit,
        #                                                           sample_limit, time_limit)
        #             elif op == "la":
        #                 result, select_idx = improver.leaf_rearrange(tree, instance, 0, list(reversed(c_branch)), assigned,
        #                                                       depth_limit, sample_limit, time_limit)
        #             else:
        #                 result, select_idx = improver.reduced_leaf(tree, instance, 0, list(reversed(c_branch)),
        #                                                              assigned,
        #                                                              depth_limit, sample_limit, time_limit)
        #             if result:
        #                 process_change(op)
        #                 # TODO: This could be more efficient...
        #                 assigned = tree.assign_samples(instance)
        #             else:
        #                 print("None")
        #             select_idx = len(c_branch) - (select_idx+1)
        #             c_branch.pop()
        #             return select_idx, assigned
        #
        #     find_leaf(tree.root, tree.assign_samples(instance))
