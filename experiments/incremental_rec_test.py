import argparse
import io
import subprocess
import sys
import os
import time
import parser
from tree_node_encoding import TreeEncoding
from diagram_encoding import DecisionDiagramEncoding
from diagram_depth import DiagramDepthEncoding
from tree_depth_encoding import TreeDepthEncoding
from bdd_instance import BddInstance
import bdd_instance
import random
import sat_tools
from collections import defaultdict
import strategies.strategies as strat
from aaai_encoding import AAAIEncoding
from switching_encoding import SwitchingEncoding
from decision_tree import DecisionTree

timeout = 3600
memlimit = 2048 * 5

input_path = sys.argv[1]
enc_idx = int(sys.argv[2])
solver_idx = int(sys.argv[3])
strat_idx = int(sys.argv[4])
enable_red = True if sys.argv[5] == "1" else False
enable_init_red = True if sys.argv[6] == "1" else False

tmp_dir = "." if len(sys.argv) == 7 else sys.argv[7]
result_dir = "." if len(sys.argv) == 7 else sys.argv[8]

encodings = [
    DecisionDiagramEncoding,
    DiagramDepthEncoding,
    TreeDepthEncoding,
    TreeEncoding,
    AAAIEncoding,
    SwitchingEncoding
]

solvers = [
    sat_tools.MiniSatSolver,
    sat_tools.GlucoseSolver,
    sat_tools.CadicalSolver
]

strategies = [
    strat.RandomStrategy,
    strat.IncrementalStrategy,
    strat.RetainingStrategy,
    strat.UpdatedRetainingStrategy,
    strat.AAAI,
    strat.NewNewStrategy
]

substrat = strat.NewNewStrategy

keys = {}
for fl in os.listdir("."):
    full_name = f"./{fl}"
    if os.path.isfile(full_name) and fl.startswith("keys_"):
        with open(full_name) as key_file:
            for ln in key_file:
                cols = ln.split(";")
                key = cols[1].split(",")[0:-1]

                if cols[0] not in keys or len(keys[cols[0]]) > len(key):
                    keys[cols[0]] = key

        print(f"Processed {fl}")
keys = {k: [int(cv) for cv in v] for k, v in keys.items()}

selected_strategy = strategies[strat_idx]
solver = solvers[solver_idx]
encoding = encodings[enc_idx]

runner = sat_tools.SatRunner(encoding, solver(), base_path=tmp_dir)

done = set()
test = os.path.split(os.path.normpath(input_path))
out_file = f"results_rec_{os.path.split(os.path.normpath(input_path))[-1]}_{enc_idx}_{solver_idx}_{enable_red}_{strat_idx}_{enable_init_red}.csv"
out_file = os.path.join(result_dir, out_file)

if not os.path.exists(out_file):
    with open(out_file, "w") as of:
        of.write("Instance;Training Acc.;Test Acc.;Nodes;Depth;Extended")
        of.write(os.linesep)

with open(out_file, "r+") as of:
    for i, ln in enumerate(of):
        if i > 0:
            lnc = ln.split(";")
            done.add(lnc[0])

    for fl in os.listdir(input_path):
        if fl.endswith(".csv"):
            if fl.endswith("_training.csv"):
                instance_name = fl[0:-1 * len("_training.csv")]
                training_instance = fl
                test_instance_name = instance_name + "_test.csv"
            else:
                instance_name = fl[0:-4]
                training_instance = fl
                test_instance_name = fl

            if instance_name in done:
                continue

            print("Starting "+instance_name)
            start_time = time.time()
            instance = parser.parse(os.path.join(input_path, training_instance))
            test_instance = parser.parse(os.path.join(input_path, test_instance_name))
            if enable_init_red:
                bdd_instance.reduce(instance, min_key=keys[training_instance])

            def find_tree(c_instance, run_infinite):
                best_tree = None
                best_instance = None
                best_acc = 0
                best_extended = sys.maxsize
                best_reduced_tree = None
                last_reduced_tree = None

                last_tree = None
                last_instance = None
                improved = False
                strategy = selected_strategy(c_instance) if run_infinite else substrat(c_instance)
                tree_cnt = 0

                target = encoding.max_instances(c_instance.num_features, 1)

                while (run_infinite or tree_cnt < 1) and best_acc < 0.99999 and (time.time() - start_time) < timeout:

                    # Increment number after all 10 computed trees
                    if tree_cnt > 0 and tree_cnt % 10 == 0:
                        target += max(10, target // 10)

                    new_instance = strategy.find_next(best_reduced_tree, last_reduced_tree, last_instance, target, improved, best_instance)

                    new_instance.check_consistency()
                    if run_infinite:
                        print(f"Using {len(new_instance.examples)} examples")
                    if enable_red:
                        bdd_instance.reduce(new_instance)
                        new_instance.check_consistency()

                    last_tree, _ = runner.run(new_instance, encoding.new_bound(last_tree, new_instance),
                                       timeout=timeout - (time.time() - start_time), memlimit=memlimit)

                    improved = False

                    if last_tree is not None:
                        if enable_red:
                            new_instance.unreduce_instance(last_tree)
                        acc = 0
                        if run_infinite:
                            acc = last_tree.get_accuracy(instance.examples)
                            last_reduced_tree = last_tree.copy()
                            last_instance = new_instance

                            if acc > best_acc:
                                best_acc = acc
                                best_reduced_tree, last_reduced_tree_tree = last_reduced_tree, best_reduced_tree
                                best_instance, last_instance = last_instance, best_instance
                                improved = True

                        # if enable_init_red and run_infinite:
                        #     c_instance.unreduce_instance(last_tree, only_tree=True)
                        #     c_instance = test_instance

                        last_tree.check_consistency()
                        tree_cnt += 1

                        # Compute extended tree
                        leafs = defaultdict(list)
                        for e in c_instance.examples:
                            # get leaf
                            pth = last_tree.get_path(e.features)[-1]
                            leafs[pth.id].append(e)

                        extension_failed = False
                        extended_sanity = -1
                        extended_depth = -1

                        for lf, st in leafs.items():
                            failed = False
                            # Check for any wrong classifications
                            c_lf = last_tree.nodes[lf]
                            for e in st:
                                if e.cls != c_lf.cls:
                                    failed = True
                                    if len(st) == len(c_instance.examples):
                                        # In this case the subtree does not subdivide anymore, prevent infinite recursion
                                        return None, None, None
                                    break

                            if failed:
                                sub_instance = BddInstance()
                                [sub_instance.add_example(e.copy()) for e in st]
                                extension, _, _ = find_tree(sub_instance, False)
                                if extension is None:
                                    extension_failed = True
                                    break

                                def apply_extension(cn, parent, polarity, n_id):
                                    if n_id is None:
                                        last_tree.nodes.append(None)
                                        n_id = len(last_tree.nodes) - 1
                                    if cn.is_leaf:
                                        last_tree.nodes.append(None)
                                        last_tree.add_leaf(n_id, parent, polarity, cn.cls)
                                    else:
                                        last_tree.add_node(n_id, parent, cn.feature, polarity)
                                        apply_extension(cn.left, n_id, True, None)
                                        apply_extension(cn.right, n_id, False, None)

                                # Find parent:
                                c_parent = None
                                c_polarity = None
                                last_tree.nodes[lf] = None

                                # Find parent
                                for ci in range(1, c_lf.id):
                                    if last_tree.nodes[ci] and not last_tree.nodes[ci].is_leaf:
                                        if last_tree.nodes[ci].left.id == c_lf.id:
                                            last_tree.nodes[ci].left = None
                                            c_parent = ci
                                            c_polarity = True
                                            break
                                        elif last_tree.nodes[ci].right.id == c_lf.id:
                                            last_tree.nodes[ci].right = None
                                            c_parent = ci
                                            c_polarity = False
                                            break
                                apply_extension(extension.root, c_parent, c_polarity, c_lf.id)

                        if not extension_failed:
                            extended_sanity = last_tree.get_accuracy(instance.examples)
                            extended_depth = last_tree.get_depth()
                            if extended_depth < best_extended:
                                best_tree = last_tree
                                best_extended = extended_depth

                        if run_infinite:
                            print(
                                f"Tree found: {acc} Accuracy,  Nodes: {last_tree.get_nodes()}, Depth: {last_tree.get_depth()},"
                                f"{extended_depth} Extended, {extended_sanity} Extended Sanity")

                return best_tree, best_acc, best_extended

            final_tree, final_acc, final_extended = find_tree(instance, True)
            if final_tree is None:
                of.write(f"{instance_name};None{os.linesep}")
                print(
                    f"Finished {instance_name}, No tree found")
            else:
                if enable_init_red:
                    instance.unreduce_instance(final_tree)
                of.write(f"{instance_name};{final_acc};{final_tree.get_accuracy(test_instance.examples)};"
                         f"{final_tree.get_nodes()};{final_tree.get_depth()};{final_extended}{os.linesep}")
                print(
                    f"Finished {instance_name}, Training accuracy: {final_acc}, Test accuracy: {final_tree.get_accuracy(test_instance.examples)}")
            of.flush()

            print("")
            print("")
