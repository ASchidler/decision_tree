import os
import nonbinary.tree_parsers as tp
from collections import defaultdict
import nonbinary.nonbinary_instance as nbi
import nonbinary.pruning as p

algos = ["w"]
trees = ["u"]
#flags = ["0", "a", "y"]
flags = ["61zu"]
#flags = ["00", "00z", "40", "40z", "50", "50z", "70", "70z"]
#flags = ["uzy"]
use_ccp = True

for c_file in sorted(os.listdir("../instances")):
    if c_file.endswith(".data"):
        for c_t in trees:
            for c_a in algos:
                fd = c_file.split(".")
                if int(fd[1]) > 1 and not os.path.exists(os.path.join("../instances", f"{fd[0]}.5.data")):
                    continue
                validation_instance, test_instance, validation_test = nbi.parse("../instances", fd[0], int(fd[1]), True, True)
                full_instance, _, _ = nbi.parse("../instances", fd[0], int(fd[1]), False, True)

                for c_f in flags:
                    out_path = os.path.join("trees", "p" if not use_ccp else "p2", f"{c_file[:-5]}.{c_t}.{c_f}.{c_a}.dt")
                    if os.path.exists(out_path):
                        continue

                    tree_path = os.path.join("trees", c_t, f"{fd[0]}.{fd[1]}.{c_t}.{c_f}.{c_a}.dt")
                    v_tree_path = os.path.join("trees", "v", f"{fd[0]}.{fd[1]}.v.v{c_f.replace('u', '')}.{c_a}.dt")
                    #v_tree_path = os.path.join("trees", "validation", f"{fd[0]}.{fd[1]}.{c_a}.dt")

                    if not os.path.exists(tree_path):
                        print("Tree does not exist: "+ tree_path)
                        continue
                    if not os.path.exists(v_tree_path):
                        print("Validation tree does not exist: "+ v_tree_path)
                        continue

                    tree = tp.parse_internal_tree(tree_path)
                    tree.root.reclassify(full_instance.examples)
                    v_tree = tp.parse_internal_tree(v_tree_path)
                    v_tree.root.reclassify(validation_instance.examples)

                    print(f"{tree_path}: {tree.get_nodes()} {tree.get_depth()} {tree.get_accuracy(test_instance.examples)}")
                    if use_ccp:
                        p.cost_complexity(tree, full_instance, v_tree, validation_instance, validation_test)
                    else:
                        p.prune_c45_optimized(tree, full_instance, v_tree, validation_instance, validation_test)
                    print(f"{tree_path}: {tree.get_nodes()} {tree.get_depth()} {tree.get_accuracy(test_instance.examples)}")

                    with open(out_path, "w") as out_file:
                        out_file.write(tree.as_string())
