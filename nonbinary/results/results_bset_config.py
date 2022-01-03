import os
import glob
import nonbinary.tree_parsers as tp
from collections import defaultdict
import nonbinary.nonbinary_instance as nbi
import nonbinary.pruning as p
import multiprocessing as mp

dirs = ["m", "k", "n", "x", "q", "f", "g", "r", "h", "i"]


def load_tree(c_fn):
    trees = list(glob.glob(os.path.join("trees", c_fn, f"{fd[0]}.{fd[1]}.*.dt")))
    if len(trees) == 0:
        print("Tree does not exist: " + c_fn + " " + f"{fd[0]}.{fd[1]}.*.dt")
        return None

    tree_path = trees[0]
    file_fields = os.path.split(tree_path)[1].split(".")
    p_tree_path = os.path.join("trees", "p", os.path.split(tree_path)[1])
    v_tree_path = os.path.join("trees", "v",
                               f"{file_fields[0]}.{file_fields[1]}.v.v{file_fields[3].replace('u', '')}.w.dt")

    if not os.path.exists(p_tree_path):
        print("Tree does not exist: " + p_tree_path)
        return None
    if not os.path.exists(v_tree_path):
        print("Validation tree does not exist: " + v_tree_path)
        return None

    tree = tp.parse_internal_tree(p_tree_path)
    tree.root.reclassify(full_instance.examples)
    v_tree = tp.parse_internal_tree(v_tree_path)
    v_tree.root.reclassify(validation_instance.examples)

    p.prune_c45_optimized(v_tree, full_instance, v_tree, validation_instance, validation_test)

    results = [v_tree.get_nodes(), v_tree.get_depth(), round(v_tree.get_accuracy(validation_test.examples), 2),
               tree.get_nodes(), tree.get_depth(), tree.get_accuracy(test_instance.examples)]
    return results

all_results = {}
cnt = 0
for c_file in sorted(os.listdir("../instances")):
    if c_file.endswith(".data"):
        cnt += 1

        fd = c_file.split(".")
        if int(fd[1]) > 1 and not os.path.exists(os.path.join("../instances", f"{fd[0]}.5.data")):
            continue
        validation_instance, test_instance, validation_test = nbi.parse("../instances", fd[0], int(fd[1]), True, True)
        full_instance, _, _ = nbi.parse("../instances", fd[0], int(fd[1]), False, True)

        c_best = None
        pool = mp.Pool(2 * mp.cpu_count())

        load_results = [pool.apply(load_tree, args=[c_f]) for c_f in dirs]
        for c_r in load_results:
            if c_r is not None:
                if c_best is None or c_best[2] < c_r[2] or (c_best[2] == c_r[2] and c_best[0] > c_r[0]):
                    c_best = c_r

        # for c_f in dirs:
        #     trees = list(glob.glob(os.path.join("trees", c_f, f"{fd[0]}.{fd[1]}.*.dt")))
        #     if len(trees) == 0:
        #         print("Tree does not exist: " + c_f + " " + f"{fd[0]}.{fd[1]}.*.dt")
        #         continue
        #
        #     tree_path = trees[0]
        #     file_fields = os.path.split(tree_path)[1].split(".")
        #     p_tree_path = os.path.join("trees", "p", os.path.split(tree_path)[1])
        #     v_tree_path = os.path.join("trees", "v", f"{file_fields[0]}.{file_fields[1]}.v.v{file_fields[3].replace('u', '')}.w.dt")
        #
        #     if not os.path.exists(p_tree_path):
        #         print("Tree does not exist: " + p_tree_path)
        #         continue
        #     if not os.path.exists(v_tree_path):
        #         print("Validation tree does not exist: " + v_tree_path)
        #         continue
        #
        #     tree = tp.parse_internal_tree(p_tree_path)
        #     tree.root.reclassify(full_instance.examples)
        #     v_tree = tp.parse_internal_tree(v_tree_path)
        #     v_tree.root.reclassify(validation_instance.examples)
        #
        #     p.prune_c45_optimized(v_tree, full_instance, v_tree, validation_instance, validation_test)
        #
        #     results = [v_tree.get_nodes(), v_tree.get_depth(), round(v_tree.get_accuracy(validation_test.examples), 2),
        #                tree.get_nodes(), tree.get_depth(), tree.get_accuracy(test_instance.examples)]
        #
        #     if c_best is None or c_best[2] < results[2] or (c_best[2] == results[2] and c_best[0] > results[0]):
        #         c_best = results

        if c_best:
            if fd[0] not in all_results:
                all_results[fd[0]] = [c_best, 1]
            else:
                c_r = all_results[fd[0]]
                for i in range(0, len(c_best)):
                    c_r[0][i] += c_best[i]
                c_r[1] += 1

        print(f"{c_best}" + os.linesep)

with open("vbest_config.csv", "w") as outp:
    outp.write("Instance;Size;Depth;Accuracy"+ os.linesep)
    for c_instance, results in all_results.items():
        results[0] = [x / results[1] for x in results[0]]
        outp.write(f"{c_instance};{results[0][3]};{results[0][4]};{results[0][5]}"+ os.linesep)
