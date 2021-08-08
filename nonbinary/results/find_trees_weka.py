import os
import shutil
import subprocess
import re
from sys import maxsize
import improve.tree_parsers as tp
import parser
from nonbinary.nonbinary_instance import ClassificationInstance, Example, parse
from collections import deque
from nonbinary.decision_tree import DecisionTree, DecisionTreeLeaf, DecisionTreeNode
from decimal import Decimal, InvalidOperation

use_validation = False
pruning = 0  # 0 is no pruning

pth = "nonbinary/instances"
weka_path = os.path.join(os.path.expanduser("~"), "Downloads/weka-3-8-5-azul-zulu-linux/weka-3-8-5")
jre_path = os.path.join(weka_path, "jre/zulu11.43.55-ca-fx-jre11.0.9.1-linux_x64/bin/java")
#-cp ./weka.jar


if pruning == 0:
    parameters = ["-U", "-J", "-M", "0"]
elif pruning == 1:
    parameters = ["-C", "0.25", "-M", "2"]

def parse_weka_tree(lines):
    wtree = DecisionTree()
    # Single leaf tree, edge case
    if lines[0].strip().startswith(":"):
        pos = lines[0].index(":")
        pos2 = lines[0].index(" ", pos+2)
        cls = lines[0][pos + 2:pos2]
        c_leaf = DecisionTreeLeaf(cls, 1, wtree)
        wtree.nodes[1] = c_leaf
        wtree.root = c_leaf
        return wtree

    c_id = 1
    l_depth = -1
    stack = []
    for ll in lines:
        depth = 0
        for cc in ll:
            if cc == " " or cc == "|":
                depth += 1
            else:
                c_line = ll[depth:].strip()
                while stack and depth < l_depth:
                    stack.pop()
                    l_depth -= 4

                cp = None if not stack else stack[-1]
                if not c_line.startswith("att"):
                    print(f"Parser error, line should start with att, starts with {c_line}.")
                    exit(1)

                if depth > l_depth:
                    feature = int(c_line[3:c_line.find(" ")])
                    t_str = c_line.split(" ")[2].replace(":", "")  # remove the : at the end
                    try:
                        threshold = int(t_str)
                    except ValueError:
                        try:
                            threshold = Decimal(t_str)
                        except InvalidOperation:
                            threshold = t_str
                    is_cat = c_line.split(" ")[1].strip() == "="
                    if cp is not None:
                        node = wtree.add_node(feature, threshold, cp.id, cp.left is None, is_cat)
                    else:
                        node = wtree.set_root(feature, threshold, is_cat)
                    stack.append(node)
                    c_id += 1
                    cp = node

                if c_line.find(":") > -1:
                    pos = c_line.index(":")
                    pos2 = c_line.index(" ", pos+2)
                    cls = c_line[pos + 2:pos2]
                    wtree.add_leaf(cls, cp.id, cp.left is None)
                    c_id += 1

                l_depth = depth
                break
    return wtree


def get_tree(instance_path, params):
    """Calls WEKA"""
    process = subprocess.Popen([
        jre_path, "-cp", os.path.join(weka_path, "weka.jar"),
        "weka.classifiers.trees.J48",
        "-t", instance_path, "-no-cv",
        *params],
        cwd=weka_path,
        #stderr=open(os.devnull, 'w'),
        stdout=subprocess.PIPE
        )

    output, _ = process.communicate()
    output = output.decode('ascii')

    mt = re.search("J48 u?n?pruned tree[^\-]*[\-]*(.*)Number of Leaves", output, re.DOTALL)


    return mt.group(1).strip().splitlines()


fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith(".data") and x.startswith("meteo")}
fls = sorted(fls)

for fl in fls:
    print(f"{fl}")
    for c_slice in range(1, 6):
        print(f"{c_slice}")

        try:
            instance, instance_test, instance_validation = parse(pth, fl, c_slice, use_validation=use_validation)
        except FileNotFoundError:
            # Invalid slice for instances with test set.
            continue

        instance.export_c45("/tmp/weka_instance.data")

        if pruning == 0:
            tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", parameters))
            with open(f"nonbinary/results/trees/unpruned/{fl}.{c_slice}.w.dt", "w") as outp:
                outp.write(tree.as_string())

#
# for fl in fls[0::5]:
#     if fl.endswith(".data"):
#         print(f"{os.linesep}Processing {fl}")
#         if pruning == 0:
#             out_fn = os.path.join(pth_out, fl[:-4] + "tree")
#         else:
#             out_fn = os.path.join(pth_out, fl[:-4] + f"p{pruning}." + "tree")
#
#         if pruning == 0:
#             if not os.path.exists(out_fn):
#                 with open(out_fn, "w") as outp:
#                     outp.write(get_tree(os.path.join(pth, fl), parameters))
#         else:
#             if os.path.exists(out_fn):
#                 continue
#
#             instance = parser.parse(os.path.join(pth, fl))
#             instance_test = parser.parse(os.path.join(pth, fl[:-4]+"test"))
#             instance_validate = None if not use_validation else parser.parse(os.path.join(pth, fl[:-4] + "validate"))
#
#             instances = []
#             if not use_validation:
#                 target_x = [x.features[1:] for x in instance.examples]
#                 target_y = [x.cls for x in instance.examples]
#                 folds = list(StratifiedKFold().split(target_x, target_y))
#
#                 for c_id, (c_fold_train, c_fold_test) in enumerate(folds):
#                     c_train = ClassificationInstance()
#                     c_test = ClassificationInstance()
#                     with open(f"/tmp/weka_instance_{c_id}_{os.getpid()}.data", "w") as tmp_file:
#                         for c_example in c_fold_train:
#                             c_train.add_example(instance.examples[c_example])
#                             tmp_file.write(",".join(f"{1 if x else 0}" for x in target_x[c_example]))
#                             tmp_file.write(f",{target_y[c_example]}"+os.linesep)
#                     shutil.copy(os.path.join(pth, fl)[:-5]+".names", f"/tmp/weka_instance_{c_id}_{os.getpid()}.names")
#
#                     for c_example in c_fold_test:
#                         c_test.add_example(instance.examples[c_example])
#
#                     instances.append((c_id, c_train, c_test, f"/tmp/weka_instance_{c_id}_{os.getpid()}.data"))
#             else:
#                 target_x = [x.features[1:] for x in instance_validate.examples]
#                 target_y = [x.cls for x in instance_validate.examples]
#                 folds = [list(range(0, len(target_x)))]
#
#                 shutil.copy(os.path.join(pth, fl)[:-5] + ".names", f"/tmp/weka_instance_0_{os.getpid()}.names")
#                 shutil.copy(os.path.join(pth, fl), f"/tmp/weka_instance_0_{os.getpid()}.data")
#                 instances.append((0, instance, instance_validate, f"/tmp/weka_instance_0_{os.getpid()}.data"))
#
#             def get_accuracy(c_val, m_val):
#                 acc = 0.0
#                 sz = 0
#                 new_params = ["-C", f"{c_val}", "-M", f"{m_val}"] if pruning == 1 else \
#                     ["-R", "-N", f"{c_val}", "-Q", "1", "-M", f"{m_val}"]
#                 for set_id, set_train, set_test, set_path in instances:
#                     c_tree = tp.parse_weka_tree(None, set_train, get_tree(set_path, new_params).split(os.linesep))
#                     acc += c_tree.get_accuracy(set_test.examples)
#                     sz += c_tree.get_nodes()
#
#                 print(f"m {m_val}, c {c_val}: {acc / len(instances)}, size {sz / len(instances)}")
#
#                 return acc, sz
#
#             # Use defaults as baseline
#             best_c = 0.25
#             best_n = 3
#             best_m = 2
#             best_accuracy, _ = get_accuracy(best_c, best_m) if pruning == 1 else get_accuracy(best_n, best_m)
#
#             if pruning == 1:
#                 c_c = 0.01
#                 while c_c < 0.5:
#                     c_accuracy, _ = get_accuracy(c_c, best_m)
#
#                     if c_accuracy > best_accuracy:
#                         best_accuracy = c_accuracy
#                         best_c = c_c
#
#                     if c_c < 0.05:
#                         c_c += 0.01
#                     else:
#                         c_c += 0.05
#             else:
#                 n_values = list(range(2, 11))
#                 for c_n in n_values:
#                     if c_n == 3:
#                         # covered in default case
#                         continue
#                     c_accuracy, _ = get_accuracy(c_n, best_m)
#                     if c_accuracy > best_accuracy:
#                         best_n = c_n
#                         best_accuracy = c_accuracy
#
#             max_m = len(instance.examples) // 5 * 4
#             m_values = [1, 2, 3, 4, *[x for x in range(5, max_m+1, 5)]]
#             c_m = 1
#             last_accuracies = deque(maxlen=5)
#
#             for c_m in m_values:
#                 c_accuracy, c_sz = get_accuracy(best_c, c_m) if pruning == 1 else get_accuracy(best_n, c_m)
#
#                 if c_accuracy > best_accuracy:
#                     best_accuracy = c_accuracy
#                     best_m = c_m
#                 elif (c_sz // len(instances) == 1) or (
#                         len(last_accuracies) >= 5 and all(x < best_accuracy for x in last_accuracies)):
#                     break
#
#                 last_accuracies.append(c_accuracy)
#
#             with open(out_fn, "w") as outp:
#                 if use_validation:
#                     fl = fl.split(".")[0] + "." + fl.split(".")[-1]
#                 if pruning == 1:
#                     final_tree = get_tree(os.path.join(main_pth, fl), ["-C", f"{best_c}", "-M", f"{best_m}"])
#                 else:
#                     final_tree = get_tree(os.path.join(main_pth, fl), ["-R", "-N", f"{best_n}", "-Q", "1", "-M", f"{best_m}"])
#                 new_tree = tp.parse_weka_tree(None, instance, final_tree.split(os.linesep))
#                 print(f"Final accuracy {new_tree.get_accuracy(instance_test.examples)}")
#                 outp.write(final_tree)
