import improve.improver as improver
import time
import sys
import decision_tree

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


def run(tree, instance, test, limit_idx=1, pt=False, timelimit=0):
    changed = True

    sample_limit = [sample_limit_short, sample_limit_mid, sample_limit_long][limit_idx]
    time_limit = time_limits[limit_idx]
    depth_limit = depth_limits[limit_idx]

    # Select nodes based on the depth
    c_ignore = set()

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

    while changed:
        changed = False

        for op in ["ma"]:
            def find_mid(root, assigned):
                # result, _ = improver.mid_rearrange(tree, instance, 0, [root], assigned, depth_limit, sample_limit, time_limit)
                result, _ = improver.mid_reduced(tree, instance, 0, [root], assigned, False, sample_limit,
                                                 depth_limit, time_limit)
                if result:
                    process_change(op)
                    # TODO: This could be more efficient...
                    assigned = tree.assign_samples(instance)
                else:
                    print("None")

                if not root.is_leaf:
                    assigned = find_mid(root.left, assigned)
                    assigned = find_mid(root.right, assigned)

                return assigned

            find_mid(tree.root, tree.assign_samples(instance))

        for op in ["ls", "la", "lr"]:
            c_branch = []

            def find_leaf(root, assigned):
                if 0 < timelimit < (time.time() - start_time):
                    return

                c_branch.append(root)
                if not root.is_leaf:
                    r1, assigned = find_leaf(root.left, assigned)
                    # root.right may become None if we reduced the sub-tree to a leaf.
                    if r1 >= len(c_branch) - 1 and root.right is not None:
                        r1, assigned = find_leaf(root.right, assigned)
                    c_branch.pop()
                    return r1, assigned
                else:
                    if op == "ls":
                        result, select_idx = improver.leaf_select(tree, instance, 0, list(reversed(c_branch)), assigned, depth_limit,
                                                                  sample_limit, time_limit)
                    elif op == "la":
                        result, select_idx = improver.leaf_rearrange(tree, instance, 0, list(reversed(c_branch)), assigned,
                                                              depth_limit, sample_limit, time_limit)
                    else:
                        result, select_idx = improver.reduced_leaf(tree, instance, 0, list(reversed(c_branch)),
                                                                     assigned,
                                                                     depth_limit, sample_limit, time_limit)
                    if result:
                        process_change(op)
                        # TODO: This could be more efficient...
                        assigned = tree.assign_samples(instance)
                    else:
                        print("None")
                    select_idx = len(c_branch) - (select_idx+1)
                    c_branch.pop()
                    return select_idx, assigned

            find_leaf(tree.root, tree.assign_samples(instance))



