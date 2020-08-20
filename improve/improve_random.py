import improve.improver as improver
import random
import time

sample_limit = 50
depth_limit = 12
reduce_runs = 1


def run(tree, instance, test):
    structure = improver.find_structure(tree)
    l_structure = list(structure.values())
    assigned = improver.assign_samples(tree, instance)
    start_time = time.time()

    def process_change():
        print(f"Time: {time.time() - start_time:.4f}\t"
              f"Training {tree.get_accuracy(instance.examples):.4f}\t"
              f"Test {tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {tree.get_depth():03}\t"
              f"Avg {tree.get_avg_depth():03.4f}\t"
              f"Nodes {tree.get_nodes()}")

    while True:
        next_node = random.randint(0, len(l_structure) - 1)
        n_n, _, n_p = l_structure[next_node]

        path = [n_n]
        while n_p is not None:
            n_n, _, n_p = structure[n_p]
            path.append(n_n)

        result, select_idx = improver.leaf_select(tree, instance, 0, path, assigned, depth_limit, sample_limit)
        if not result:
            result, idx1 = improver.leaf_rearrange(tree, instance, select_idx, path, assigned, depth_limit,
                                                  sample_limit)
        if not result:
            result, idx2 = improver.reduced_leaf(tree, instance, select_idx, path, assigned, depth_limit, sample_limit)

        if not result:
            idx1 = min(len(path) - 1, idx1 + 1)
            result, idx = improver.mid_reduced(tree, instance, idx1, path, assigned, False, depth_limit, sample_limit)

        if not result:
            idx2 = min(len(path) - 1, idx2 + 1)
            result, idx = improver.mid_reduced(tree, instance, idx2, path, assigned, True, depth_limit,
                                           sample_limit)

        if result:
            structure = improver.find_structure(tree)
            l_structure = list(structure.values())
            assigned = improver.assign_samples(tree, instance)
            process_change()
